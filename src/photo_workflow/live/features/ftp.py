"""
Local FTP server that lets a Canon camera upload RAW files into the Live folder.

Built on `pyftpdlib`. Uploads are written to a hidden staging folder inside the Live
`incoming` folder and moved into `incoming` only once the transfer completes, then handed
to the pipeline with `host.ingest`. An interrupted upload is discarded and reported as a
warning. Only the local network may connect, and the server only runs while the feature
is switched on.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
from pathlib import Path

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

from photo_workflow.live.api import StageHost, StageImplementation, StageRegistry
from photo_workflow.live.layout import RAW_SUFFIXES
from photo_workflow.live.stages import StageId

LOGGER = logging.getLogger(__name__)
STAGING_FOLDER_NAME = ".ftp-uploading"
PASSIVE_PORTS = range(60000, 60100)
PERMISSIONS = "elw"  # change directory, list, store; no delete, rename or mkdir
IDLE_TIMEOUT_SECONDS = 300
POLL_SECONDS = 0.2


def is_local_address(address: str) -> bool:
    """
    Return whether a peer address is on the local machine or a private/link-local network.

    Returns:
        `True` for loopback, private (RFC 1918 / ULA) and link-local addresses.
    """
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_private or parsed.is_loopback or parsed.is_link_local


def local_address() -> str:
    """
    Return this machine's LAN address that a camera should connect to.

    Returns:
        The IPv4 address of the default network interface, or `127.0.0.1` when offline.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # UDP connect sends no packets
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def _unique_destination(folder: Path, name: str) -> Path:
    destination = folder / name
    counter = 1
    while destination.exists():
        destination = folder / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    return destination


class FtpServer:
    """A pyftpdlib server delivering completed RAW uploads to a `StageHost`."""

    def __init__(self, host: StageHost) -> None:
        """
        Prepare the server from the host's current Live settings.

        Raises:
            ValueError: If the FTP username or password is not configured.
        """
        settings = host.settings
        if not settings.ftp_username or not settings.ftp_password:
            raise ValueError("Set the FTP username and password in Preferences → Live")
        self._host = host
        self._incoming = host.layout.incoming
        self._staging = self._incoming / STAGING_FOLDER_NAME
        self._port = settings.ftp_port
        self._username = settings.ftp_username
        self._password = settings.ftp_password
        self._address = ""
        self._server: FTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def port(self) -> int:
        """The port the server listens on (the real one once started)."""
        return self._server.address[1] if self._server else self._port

    def status_text(self) -> str:
        """Return the connection details to enter on the camera."""
        address = self._address or local_address()
        return f"FTP server running at {address}:{self.port} (passive mode)"

    def start(self) -> None:
        """
        Start listening for the camera on a background thread.

        A port that cannot be bound raises `OSError`, which halts the feature.
        """
        self._address = local_address()
        self._staging.mkdir(parents=True, exist_ok=True)
        for leftover in self._staging.iterdir():
            leftover.unlink(missing_ok=True)
        authorizer = DummyAuthorizer()
        authorizer.add_user(self._username, self._password, str(self._staging), perm=PERMISSIONS)
        handler = type("CameraHandler", (_CameraHandler,), {})
        handler.authorizer = authorizer
        handler.passive_ports = PASSIVE_PORTS
        handler.timeout = IDLE_TIMEOUT_SECONDS
        handler.banner = "photo-workflow FTP ready"
        handler.owner = self
        self._server = FTPServer(("0.0.0.0", self._port), handler)  # noqa: S104
        self._server.max_cons = 16
        self._server.max_cons_per_ip = 8
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="live-ftp", daemon=True)
        self._thread.start()
        LOGGER.info(self.status_text())

    def stop(self) -> None:
        """Stop the server, dropping any upload still in progress."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._server is not None:
            self._server.close_all()
        self._server = None
        self._thread = None
        LOGGER.info("FTP server stopped")

    def _serve(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            while not self._stop.is_set():
                server.serve_forever(timeout=POLL_SECONDS, blocking=False, handle_exit=False)
        except Exception as error:
            LOGGER.exception("FTP server crashed")
            self._host.fail(StageId.FTP, f"FTP server stopped unexpectedly: {error}")

    def file_received(self, path: Path) -> None:
        """Move a completed upload into the landing folder and hand it to the pipeline."""
        if path.suffix.lower() not in RAW_SUFFIXES:
            LOGGER.info("ignoring non-RAW upload %s", path.name)
            path.unlink(missing_ok=True)
            return
        destination = _unique_destination(self._incoming, path.name)
        os.replace(path, destination)
        LOGGER.info("upload complete: %s", destination.name)
        self._host.ingest(destination)

    def file_interrupted(self, path: Path) -> None:
        """Discard a partial upload and report it as a warning."""
        path.unlink(missing_ok=True)
        if path.suffix.lower() in RAW_SUFFIXES:
            self._host.warn(
                StageId.FTP,
                f"Upload of {path.name} was interrupted; send the photo again.",
            )


class _CameraHandler(FTPHandler):
    """FTP session handler that only serves the local network."""

    owner: FtpServer

    def on_connect(self) -> None:
        """Drop connections from outside the local network."""
        if not is_local_address(self.remote_ip):
            LOGGER.warning("rejected FTP connection from non-local address %s", self.remote_ip)
            self.close()

    def on_file_received(self, file: str) -> None:
        """Hand a fully uploaded file to the server."""
        self.owner.file_received(Path(file))

    def on_incomplete_file_received(self, file: str) -> None:
        """Report an interrupted upload."""
        self.owner.file_interrupted(Path(file))


class FtpStage(StageImplementation):
    """Runs the FTP server while the FTP feature is on."""

    def __init__(self) -> None:
        """Start with no server running."""
        self._server: FtpServer | None = None

    @property
    def server(self) -> FtpServer | None:
        """The running server, or `None` when the feature is off."""
        return self._server

    def status_text(self) -> str:
        """Return the address and port to enter on the camera while the server runs."""
        return self._server.status_text() if self._server is not None else ""

    def start(self, host: StageHost) -> None:
        """Start the server (restarting it if already running)."""
        self.stop()
        server = FtpServer(host)
        server.start()
        self._server = server

    def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.stop()
            self._server = None


def register(registry: StageRegistry) -> None:
    """Register the FTP feature."""
    registry.register(StageId.FTP, FtpStage())
