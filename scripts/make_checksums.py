#!/usr/bin/env python3
"""Regenerate the portable SHA-256 manifest for public release files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_checksums.sha256"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml", ".cff"}


def portable_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
        data = data.replace(b"\r\n", b"\n")
    return data


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path != MANIFEST
        and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(portable_bytes(path)).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} checksums to {MANIFEST.name}")


if __name__ == "__main__":
    main()
