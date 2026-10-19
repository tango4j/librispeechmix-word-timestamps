"""Verify released files against data/MANIFEST.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import sha256


def verify(data_dir: Path) -> None:
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {expected['sha256']}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    verify(args.data_dir)
    print(f"Verified {(args.data_dir / 'MANIFEST.json').resolve()}")


if __name__ == "__main__":
    main()
