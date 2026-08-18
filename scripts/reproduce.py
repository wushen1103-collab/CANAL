#!/usr/bin/env python3
"""Portable entry point for the complete CANAL reproduction workflow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SPLITS = ["random", "scaffold", "source", "temporal", "joint"]
SEEDS = ["13", "29", "47", "61", "83"]
CHEMBL = Path("data/processed/chembl37_canal20.parquet")
BINDINGDB_ALL = Path("data/processed/bindingdb_articles_202608_canal20.parquet")
BINDINGDB_STRICT = Path("data/processed/bindingdb_articles_202608_independent_strict.parquet")


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def execute(command: list[str], workers: int, dry_run: bool) -> None:
    printable = shlex.join(command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    thread_count = str(max(1, min(workers, 16)))
    env.setdefault("OMP_NUM_THREADS", thread_count)
    env.setdefault("MKL_NUM_THREADS", thread_count)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def prepare(args: argparse.Namespace) -> None:
    if args.chembl_sqlite is None or args.bindingdb_zip is None:
        raise SystemExit("prepare/all requires --chembl-sqlite and --bindingdb-zip")
    execute(
        [
            PYTHON,
            script("prepare_chembl.py"),
            "--sqlite",
            str(args.chembl_sqlite),
            "--out",
            str(CHEMBL),
            "--workers",
            str(args.workers),
        ],
        args.workers,
        args.dry_run,
    )
    execute(
        [
            PYTHON,
            script("prepare_bindingdb.py"),
            "--zip",
            str(args.bindingdb_zip),
            "--out",
            str(BINDINGDB_ALL),
            "--chembl-reference",
            str(CHEMBL),
            "--workers",
            str(args.workers),
        ],
        args.workers,
        args.dry_run,
    )
    execute(
        [
            PYTHON,
            script("audit_results.py"),
            "--chembl",
            str(CHEMBL),
            "--bindingdb",
            str(BINDINGDB_ALL),
            "--out",
            "artifacts/audits",
            "--overlap-only",
        ],
        args.workers,
        args.dry_run,
    )


def benchmark_command(data: Path, out: str, method_set: str, args: argparse.Namespace, *, semisynthetic: bool = False, quick: bool = False, permute: bool = False) -> list[str]:
    command = [
        PYTHON,
        script("run_benchmark.py"),
        "--data",
        str(data),
        "--artifacts-root",
        out,
        "--splits",
        *SPLITS,
        "--seeds",
        *SEEDS,
        "--workers",
        str(args.workers),
        "--n-bits",
        "1024",
        "--device",
        args.device,
        "--method-set",
        method_set,
    ]
    if method_set in {"trees", "calibrated"} or permute:
        command.extend(["--tree-preset", "fast"])
    if semisynthetic:
        command.append("--semisynthetic")
    if quick:
        command.append("--quick")
    if permute:
        command.extend(["--permute-metadata-seed", "20260813"])
    return command


def run_benchmark(args: argparse.Namespace) -> None:
    jobs = [
        benchmark_command(CHEMBL, "artifacts/chembl_core", "core", args, semisynthetic=True, quick=True),
        benchmark_command(CHEMBL, "artifacts/chembl_trees", "trees", args, semisynthetic=True),
        benchmark_command(CHEMBL, "artifacts/chembl_calibrated", "calibrated", args, semisynthetic=True),
        benchmark_command(CHEMBL, "artifacts/chembl_meta_perm", "core", args, semisynthetic=True, quick=True, permute=True),
        benchmark_command(BINDINGDB_ALL, "artifacts/bindingdb_all_core", "core", args, semisynthetic=True, quick=True),
        benchmark_command(BINDINGDB_ALL, "artifacts/bindingdb_all_calibrated", "calibrated", args, semisynthetic=True),
    ]
    for command in jobs:
        execute(command, args.workers, args.dry_run)
    if not args.skip_smiles:
        execute(
            [
                PYTHON,
                script("run_smiles_cnn.py"),
                "--data",
                str(CHEMBL),
                "--artifacts-root",
                "artifacts/chembl_smiles_cnn",
                "--splits",
                *SPLITS,
                "--seeds",
                *SEEDS,
                "--modes",
                "censored",
                "--device",
                args.device,
                "--batch-size",
                "1536",
                "--epochs",
                "32",
                "--patience",
                "5",
                "--semisynthetic",
            ],
            args.workers,
            args.dry_run,
        )
    inputs = [
        "artifacts/chembl_core",
        "artifacts/chembl_trees",
        "artifacts/chembl_calibrated",
        "artifacts/chembl_meta_perm",
        "artifacts/bindingdb_all_core",
        "artifacts/bindingdb_all_calibrated",
    ]
    if not args.skip_smiles:
        inputs.append("artifacts/chembl_smiles_cnn")
    execute(
        [PYTHON, script("summarize_results.py"), "--inputs", *inputs, "--out", "artifacts/combined"],
        args.workers,
        args.dry_run,
    )


def run_diagnostics(args: argparse.Namespace) -> None:
    execute(
        [
            PYTHON,
            script("analyze_paired_calibration.py"),
            "--data",
            str(CHEMBL),
            "--out",
            "artifacts/paired_calibration",
            "--workers",
            str(args.workers),
            "--n-bits",
            "1024",
        ],
        args.workers,
        args.dry_run,
    )
    execute(
        [
            PYTHON,
            script("analyze_targetwise_calibration.py"),
            "--data",
            str(CHEMBL),
            "--out",
            "artifacts/targetwise_calibration",
            "--workers",
            str(args.workers),
            "--n-bits",
            "1024",
        ],
        args.workers,
        args.dry_run,
    )
    execute(
        [
            PYTHON,
            script("audit_results.py"),
            "--chembl",
            str(CHEMBL),
            "--bindingdb",
            str(BINDINGDB_ALL),
            "--metrics",
            "artifacts/combined/metrics_long.csv",
            "--out",
            "artifacts/audits",
        ],
        args.workers,
        args.dry_run,
    )


def run_external(args: argparse.Namespace) -> None:
    execute(benchmark_command(BINDINGDB_STRICT, "artifacts/bindingdb_independent_strict_core", "core", args, quick=True), args.workers, args.dry_run)
    execute(benchmark_command(BINDINGDB_STRICT, "artifacts/bindingdb_independent_strict_calibrated", "calibrated", args), args.workers, args.dry_run)
    execute(
        [
            PYTHON,
            script("summarize_results.py"),
            "--inputs",
            "artifacts/bindingdb_independent_strict_core",
            "artifacts/bindingdb_independent_strict_calibrated",
            "--out",
            "artifacts/external_strict",
        ],
        args.workers,
        args.dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "benchmark", "diagnostics", "external", "all"], default="all")
    parser.add_argument("--chembl-sqlite", type=Path, help="Extracted ChEMBL 37 SQLite database")
    parser.add_argument("--bindingdb-zip", type=Path, help="BindingDB article-derived TSV archive, release 202608")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda:0")
    parser.add_argument("--skip-smiles", action="store_true", help="Skip the SMILES-CNN route; this does not reproduce the complete 15-method grid")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"benchmark", "all"}:
        run_benchmark(args)
    if args.stage in {"diagnostics", "all"}:
        run_diagnostics(args)
    if args.stage in {"external", "all"}:
        run_external(args)


if __name__ == "__main__":
    main()
