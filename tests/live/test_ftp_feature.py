"""Tests for the Live FTP server, using a real FTP client against a real local server."""

from __future__ import annotations

import ftplib
import io
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from photo_workflow.live.api import StageRegistry
from photo_workflow.live.features.ftp import (
    STAGING_FOLDER_NAME,
    FtpServer,
    FtpStage,
    is_local_address,
    register,
)
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.pipeline import LivePipeline
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId

USER = "camera"
PASSWORD = "s3cret"  # noqa: S105
PAYLOAD = b"RAWDATA" * 50_000


class FakeHost:
    """Records what the FTP feature hands to the pipeline."""

    def __init__(self, root: Path, **settings: object) -> None:
        """Create a host whose settings enable credentialed FTP on a free port."""
        self.layout = LiveLayout(root)
        self.settings = LiveSettings(
            ftp_port=0, ftp_username=USER, ftp_password=PASSWORD, **settings
        )
        self.ingested: list[Path] = []
        self.contents: list[bytes] = []
        self.warnings: list[tuple[StageId, str]] = []
        self.failures: list[str] = []

    def ingest(self, path: Path) -> None:
        """Record a hand-over, capturing content before anything could move the file."""
        self.ingested.append(path)
        self.contents.append(path.read_bytes())

    def warn(self, stage: StageId, message: str, photo: Path | None = None) -> None:
        """Record a warning."""
        self.warnings.append((stage, message))

    def fail(self, stage: StageId, message: str) -> None:
        """Record a halt."""
        self.failures.append(message)


@pytest.fixture
def host(tmp_path: Path) -> FakeHost:
    """Return a fake host rooted in a temporary Live folder."""
    return FakeHost(tmp_path / "live")


@pytest.fixture
def server(host: FakeHost) -> Iterator[FtpServer]:
    """
    Run an FTP server for the host and stop it afterwards.

    Yields:
        The running server.
    """
    ftp_server = FtpServer(host)
    ftp_server.start()
    yield ftp_server
    ftp_server.stop()


def connect(server: FtpServer, password: str = PASSWORD) -> ftplib.FTP:
    """
    Log in like the camera does (passive mode).

    Returns:
        A logged-in client.
    """
    client = ftplib.FTP()  # noqa: S321
    client.connect("127.0.0.1", server.port, timeout=10)
    client.set_pasv(True)
    client.login(USER, password)
    return client


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """
    Wait until a condition holds.

    Raises:
        AssertionError: If it does not hold within the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_completed_upload_is_ingested_from_incoming(host: FakeHost, server: FtpServer) -> None:
    """A finished upload lands in incoming and is handed to the pipeline once."""
    client = connect(server)
    client.storbinary("STOR IMG_0001.CR3", io.BytesIO(PAYLOAD))
    client.quit()

    wait_for(lambda: bool(host.ingested))
    assert host.ingested == [host.layout.incoming / "IMG_0001.CR3"]
    assert host.contents == [PAYLOAD]
    assert list((host.layout.incoming / STAGING_FOLDER_NAME).iterdir()) == []
    assert host.warnings == []


def test_name_collision_in_incoming_keeps_both_files(host: FakeHost, server: FtpServer) -> None:
    """Two pending uploads with the same camera name do not overwrite each other."""
    for _ in range(2):
        client = connect(server)
        client.storbinary("STOR IMG_0001.CR2", io.BytesIO(PAYLOAD))
        client.quit()
    wait_for(lambda: len(host.ingested) == 2)
    assert len({path.name for path in host.ingested}) == 2


def test_interrupted_upload_warns_and_is_not_ingested(host: FakeHost, server: FtpServer) -> None:
    """A dropped connection mid-upload is a warning and leaves no file behind."""
    client = connect(server)
    data = client.transfercmd("STOR IMG_0002.CR3")
    data.sendall(PAYLOAD[:1000])
    time.sleep(0.2)
    client.sock.shutdown(socket.SHUT_RDWR)  # the control connection drops mid-transfer
    time.sleep(0.5)
    data.close()

    wait_for(lambda: bool(host.warnings))
    assert host.ingested == []
    assert "IMG_0002.CR3" in host.warnings[0][1]
    assert host.warnings[0][0] is StageId.FTP
    assert list((host.layout.incoming / STAGING_FOLDER_NAME).iterdir()) == []
    assert [p for p in host.layout.incoming.iterdir() if p.is_file()] == []


def test_non_raw_upload_is_discarded(host: FakeHost, server: FtpServer) -> None:
    """Only CR2/CR3 files are accepted for ingestion."""
    client = connect(server)
    client.storbinary("STOR IMG_0003.JPG", io.BytesIO(b"jpeg"))
    client.storbinary("STOR IMG_0003.CR3", io.BytesIO(b"raw"))
    client.quit()

    wait_for(lambda: bool(host.ingested))
    assert [path.name for path in host.ingested] == ["IMG_0003.CR3"]
    assert not (host.layout.incoming / "IMG_0003.JPG").exists()


def test_wrong_password_is_rejected(server: FtpServer) -> None:
    """Bad credentials cannot log in."""
    with pytest.raises(ftplib.error_perm):
        connect(server, password="wrong")


def test_client_cannot_delete_or_make_directories(server: FtpServer) -> None:
    """The camera account may only upload."""
    client = connect(server)
    client.storbinary("STOR IMG_0004.CR3", io.BytesIO(PAYLOAD))
    with pytest.raises(ftplib.error_perm):
        client.delete("IMG_0004.CR3")
    with pytest.raises(ftplib.error_perm):
        client.mkd("evil")
    client.quit()


def test_server_stops_listening_when_stopped(host: FakeHost) -> None:
    """Nothing listens once the feature is off."""
    ftp_server = FtpServer(host)
    ftp_server.start()
    port = ftp_server.port
    ftp_server.stop()
    with pytest.raises(OSError):
        ftplib.FTP().connect("127.0.0.1", port, timeout=2)  # noqa: S321


def test_missing_credentials_prevent_start(tmp_path: Path) -> None:
    """The server refuses to start without credentials."""
    host = FakeHost(tmp_path / "live")
    host.settings = LiveSettings(ftp_port=0)
    with pytest.raises(ValueError, match="username and password"):
        FtpServer(host)


def test_stage_starts_and_stops_server_and_is_registered(host: FakeHost) -> None:
    """The stage manages the server lifecycle and registers itself."""
    stage = FtpStage()
    assert stage.server is None
    stage.start(host)
    assert stage.server is not None
    assert "passive" in stage.server.status_text()
    stage.stop()
    assert stage.server is None
    registry = StageRegistry()
    register(registry)
    assert registry.is_registered(StageId.FTP)


@pytest.mark.parametrize(
    ("address", "local"),
    [
        ("127.0.0.1", True),
        ("192.168.1.20", True),
        ("10.0.0.5", True),
        ("172.16.4.2", True),
        ("169.254.1.1", True),
        ("::1", True),
        ("::ffff:192.168.1.20", True),
        ("fe80::1%en0", True),
        ("8.8.8.8", False),
        ("2606:4700::1111", False),
        ("not-an-ip", False),
    ],
)
def test_is_local_address(address: str, local: bool) -> None:
    """Only loopback, private and link-local peers count as local."""
    assert is_local_address(address) is local


def test_upload_reaches_pipeline_input_folder(tmp_path: Path) -> None:
    """An upload through a real pipeline ends up renamed in the date folder's input."""
    registry = StageRegistry()
    register(registry)
    layout = LiveLayout(tmp_path / "live")
    settings = LiveSettings(ftp_port=0, ftp_username=USER, ftp_password=PASSWORD)
    pipeline = LivePipeline(layout, settings, registry)
    try:
        pipeline.toggle(StageId.FTP, True)
        server = registry.get(StageId.FTP).server
        assert server is not None
        client = connect(server)
        client.storbinary("STOR IMG_0005.CR3", io.BytesIO(PAYLOAD))
        client.quit()

        wait_for(lambda: bool(layout.list_photos()))
        assert layout.list_incoming() == []
        assert layout.list_photos()[0].read_bytes() == PAYLOAD
        pipeline.toggle(StageId.FTP, False)
        assert registry.get(StageId.FTP).server is None
    finally:
        pipeline.shutdown()
