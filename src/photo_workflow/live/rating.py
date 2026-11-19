"""Reading the star rating from an XMP sidecar."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from contextlib import suppress
from pathlib import Path

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)


def read_rating(xmp_path: Path) -> int:
    """
    Read the star rating from a sidecar; a missing or unreadable sidecar counts as 0.

    Returns:
        The rating from 0 to 5 (a rejected photo, -1, counts as 0).
    """
    try:
        data = xmp_path.read_bytes()
    except OSError:
        return 0
    packet = XMP_PACKET.search(data)
    if packet is None:
        return 0
    try:
        root = ElementTree.fromstring(packet.group(0))
    except ElementTree.ParseError:
        return 0
    rating = 0
    for description in root.iter(f"{{{RDF_NS}}}Description"):
        value = description.get(f"{{{XMP_NS}}}Rating")
        if value is None:
            element = description.find(f"{{{XMP_NS}}}Rating")
            value = element.text if element is not None else None
        if value:
            with suppress(ValueError):
                rating = max(rating, int(float(value)))
    return min(max(rating, 0), 5)
