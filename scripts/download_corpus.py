"""Download the sample textbook (OpenStax Principles of Data Science) into data/corpus/."""

from __future__ import annotations

import sys
from pathlib import Path

from coursepilot.sample import ChecksumError, download_sample

DEST = Path(__file__).resolve().parents[1] / "data" / "corpus"


def main() -> int:
    try:
        path = download_sample(DEST)
    except ChecksumError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Saved and verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
