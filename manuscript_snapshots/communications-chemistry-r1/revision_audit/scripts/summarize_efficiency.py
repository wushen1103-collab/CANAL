#!/usr/bin/env python
"""Create a publication-ready efficiency summary from monitored benchmark runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch


def elapsed_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    raise ValueError(value)


def parse_time(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.strip().split(": ", 1)
            fields[key] = value
    elapsed_key = next(key for key in fields if key.startswith("Elapsed (wall clock)"))
    return {
        "command": fields["Command being timed"].strip('"'),
        "wall_seconds": elapsed_seconds(fields[elapsed_key]),
        "max_rss_gib": int(fields["Maximum resident set size (kbytes)"]) / 1024**2,
        "mean_cpu_percent": int(fields["Percent of CPU this job got"].rstrip("%")),
        "exit_status": int(fields["Exit status"]),
    }


def parse_gpu(path: Path) -> tuple[int, int]:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return int(fields["peak_gpu_memory_mib"]), int(fields["samples"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/efficiency.csv"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    canal_time = parse_time(repo / "reports/timing_efficiency_canal_s1_s2027_20260903.txt")
    rows.append(
        {
            "method": "CANAL",
            "device": "32 CPU threads",
            **canal_time,
            "peak_gpu_memory_mib": 0,
            "gpu_samples": 0,
            "trainable_parameters": pd.NA,
            "model_structure": "20 target-specific LightGBM models; 200 trees/model; max 63 leaves/tree",
            "preprocessing_scope": "ECFP4 and descriptor construction included",
        }
    )

    dmpnn_time = parse_time(repo / "reports/timing_efficiency_dmpnn_s1_s2027_monitored_20260903.txt")
    dmpnn_gpu, dmpnn_samples = parse_gpu(
        repo / "reports/gpu_efficiency_dmpnn_s1_s2027_monitored_20260903.txt"
    )
    dmpnn_log = (repo / "logs/efficiency_dmpnn_s1_s2027_monitored_20260903.log").read_text(
        encoding="utf-8", errors="replace"
    )
    match = re.search(r"Total params:\s+([0-9,]+) K", dmpnn_log)
    dmpnn_parameters = int(match.group(1).replace(",", "")) * 1000 if match else 330_000
    rows.append(
        {
            "method": "D-MPNN bounded MSE",
            "device": "1 RTX 4090D + 4 CPU workers",
            **dmpnn_time,
            "peak_gpu_memory_mib": dmpnn_gpu,
            "gpu_samples": dmpnn_samples,
            "trainable_parameters": dmpnn_parameters,
            "model_structure": "one 20-head multitask D-MPNN",
            "preprocessing_scope": "prepared molecular/source table reused",
        }
    )

    attentive_time = parse_time(repo / "reports/timing_efficiency_attentivefp_s1_s2027_monitored_20260903.txt")
    attentive_gpu, attentive_samples = parse_gpu(
        repo / "reports/gpu_efficiency_attentivefp_s1_s2027_monitored_20260903.txt"
    )
    checkpoint = repo / "results/modern/attentivefp_models/attentivefp_s1_bounded_mse_s2027_efficiency_monitored_20260903.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    attentive_parameters = sum(value.numel() for value in state.values())
    rows.append(
        {
            "method": "AttentiveFP bounded MSE",
            "device": "1 RTX 4090D + 4 CPU workers",
            **attentive_time,
            "peak_gpu_memory_mib": attentive_gpu,
            "gpu_samples": attentive_samples,
            "trainable_parameters": attentive_parameters,
            "model_structure": "one 20-head multitask AttentiveFP",
            "preprocessing_scope": "prepared graph cache reused",
        }
    )

    frame = pd.DataFrame(rows)
    if not frame["exit_status"].eq(0).all():
        raise RuntimeError("At least one efficiency benchmark failed")
    frame.to_csv(output, index=False)
    manifest = {
        "workload": "ChEMBL35 IC50, S1, seed 2027, all 20 targets, fit plus validation/test evaluation",
        "timing": "GNU time wall clock and maximum resident set size",
        "gpu_monitoring": "nvidia-smi process-memory sum sampled every second on an otherwise idle assigned GPU",
        "comparability_note": "CANAL includes molecular feature construction; neural runs reuse their published prepared inputs/caches",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

