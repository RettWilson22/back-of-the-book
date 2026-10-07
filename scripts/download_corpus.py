"""Download the sample textbook used by the demo and the evaluation.

OpenStax "Principles of Data Science" (CC BY-NC-SA 4.0). The PDF is not committed to this
repository; this script fetches it from OpenStax and verifies it is the exact edition the
evaluation's page labels were written against.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Data-Science-WEB.pdf"
SHA256 = "6b47205459a2f4e25a2b2a86369e674a26a5ae566a4cc428dd31047670692398"
DEST = Path(__file__).resolve().parents[1] / "data" / "corpus" / "principles-of-data-science.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"Already downloaded: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    actual = sha256(DEST)
    if actual != SHA256:
        print(
            f"Warning: checksum {actual} does not match the edition used for the evaluation.\n"
            "OpenStax may have published an update; eval page labels could be off.",
            file=sys.stderr,
        )
        return 1
    print(f"Saved and verified: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
