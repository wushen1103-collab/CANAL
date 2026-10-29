#!/usr/bin/env python3
"""Freeze a hash-addressed code and result snapshot for the Reviewer 2 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/reviewer2_validation_2026-10-28"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    freeze_path = report_dir / "environment_pip_freeze.txt"
    freeze_path.write_text(
        subprocess.check_output([str(root / ".conda" / "bin" / "python"), "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    modern_freeze_path = report_dir / "environment_modern_pip_freeze.txt"
    modern_freeze_path.write_text(
        subprocess.check_output(
            [str(root / ".conda-modern" / "bin" / "python"), "-m", "pip", "freeze"],
            text=True,
        ),
        encoding="utf-8",
    )
    archive_path = report_dir / "reviewer2_code_snapshot_2026-10-28.tar.gz"
    archive_members: list[Path] = [
        root / "README.md",
        root / "EXPERIMENT_LOG.md",
        root / "requirements.txt",
    ]
    archive_members.extend(sorted((root / "scripts").glob("*.py")))
    archive_members.extend(sorted((root / "scripts").glob("*.sh")))
    archive_members.extend(
        path
        for path in [
            root / "vendor/uq4dd/config/db/descriptors/handcrafted.yaml",
            root / "vendor/uq4dd/config/model/predictor/mve.yaml",
            root / "vendor/uq4dd/config/model/optimizer/default.yaml",
            root / "vendor/uq4dd/config/model/scheduler/default.yaml",
            root / "vendor/uq4dd/uq4dd/utils/loss_functions.py",
            root / "vendor/uq4dd/uq4dd/model/predictor/mlp.py",
        ]
        if path.exists()
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in archive_members:
            if path.exists():
                archive.add(path, arcname=path.relative_to(root))

    status = git(root, "status", "--short")
    report_files = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "reviewer2_snapshot_manifest.json":
            report_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "git_head": git(root, "rev-parse", "HEAD"),
        "git_status_clean": not bool(status),
        "git_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(
            git(root, "diff", "--binary").encode("utf-8")
        ).hexdigest(),
        "warning": (
            "The working tree is not clean. This hash-addressed archive freezes the audit code, "
            "but a clean manuscript-release commit/tag is still required before public release."
            if status
            else "The working tree was clean when the snapshot was created."
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment_pip_freeze_sha256": sha256(freeze_path),
        "environment_modern_pip_freeze_sha256": sha256(modern_freeze_path),
        "code_archive": {
            "path": str(archive_path.relative_to(root)),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        },
        "report_files": report_files,
    }
    manifest_path = report_dir / "reviewer2_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ["git_head", "git_status_clean", "tracked_diff_sha256", "code_archive"]}, indent=2))


if __name__ == "__main__":
    main()
