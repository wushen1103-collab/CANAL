#!/usr/bin/env python
"""Reviewer 2 audits and prediction-only reanalyses for CANAL.

This script never edits manuscript sources. It writes machine-readable evidence to a
dedicated report directory and uses only frozen datasets and row-level predictions.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import r2_score


TREATMENT_LABELS = {
    "exact_only": "Exact-only LightGBM",
    "censored_as_exact": "Censored-as-exact LightGBM",
    "censored_likelihood": "CANAL",
}
SPLITS = ["S1", "S2", "S3", "S4"]
SEEDS = [2027, 2039, 2053, 2069, 2081]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def stable_censored_nll(
    threshold: np.ndarray, relation: np.ndarray, pred: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.05)
    z = (np.asarray(threshold, dtype=float) - np.asarray(pred, dtype=float)) / sigma
    relation = np.asarray(relation).astype(str)
    return np.where(relation == "<", -log_ndtr(z), -log_ndtr(-z))


def violation_distance(threshold: np.ndarray, relation: np.ndarray, pred: np.ndarray) -> np.ndarray:
    threshold = np.asarray(threshold, dtype=float)
    pred = np.asarray(pred, dtype=float)
    relation = np.asarray(relation).astype(str)
    return np.where(relation == "<", np.maximum(pred - threshold, 0.0), np.maximum(threshold - pred, 0.0))


def cell_metrics(frame: pd.DataFrame, pred: np.ndarray | None = None) -> dict[str, float | int]:
    prediction = frame["pred"].to_numpy(dtype=float) if pred is None else np.asarray(pred, dtype=float)
    relation = frame["p_relation"].astype(str).to_numpy()
    truth = frame["p_activity"].to_numpy(dtype=float)
    exact = relation == "="
    censored = ~exact
    result: dict[str, float | int] = {
        "exact_n": int(exact.sum()),
        "censored_n": int(censored.sum()),
    }
    if exact.sum():
        residual = prediction[exact] - truth[exact]
        result["exact_mae"] = float(np.mean(np.abs(residual)))
        result["exact_rmse"] = float(np.sqrt(np.mean(residual**2)))
        result["exact_r2"] = float(r2_score(truth[exact], prediction[exact])) if exact.sum() > 1 else np.nan
        result["exact_spearman"] = float(spearmanr(truth[exact], prediction[exact]).statistic)
        result["exact_kendall"] = float(kendalltau(truth[exact], prediction[exact]).statistic)
    else:
        for key in ("exact_mae", "exact_rmse", "exact_r2", "exact_spearman", "exact_kendall"):
            result[key] = np.nan
    if censored.sum():
        sigma = frame["sigma"].to_numpy(dtype=float)[censored]
        nll = stable_censored_nll(truth[censored], relation[censored], prediction[censored], sigma)
        compatible = np.where(
            relation[censored] == "<",
            prediction[censored] < truth[censored],
            prediction[censored] > truth[censored],
        )
        result["censored_nll_mean"] = float(np.mean(nll))
        result["censored_nll_median"] = float(np.median(nll))
        result["cdir"] = float(np.mean(compatible))
        result["violation_distance"] = float(
            np.mean(violation_distance(truth[censored], relation[censored], prediction[censored]))
        )
    else:
        for key in ("censored_nll_mean", "censored_nll_median", "cdir", "violation_distance"):
            result[key] = np.nan
    return result


def load_predictions(patterns: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        files.extend(matches if matches else [Path(pattern)])
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    chunks = []
    for path in files:
        chunk = pd.read_parquet(path)
        if "partition" not in chunk:
            chunk["partition"] = "test"
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    return frame, files


def audit_relations_and_splits(dataset: pd.DataFrame, output: Path) -> dict[str, object]:
    relation = (
        dataset.groupby(["target_chembl_id", "p_relation"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["=", "<", ">"], fill_value=0)
        .reset_index()
        .rename(columns={"=": "exact", "<": "left_censored", ">": "right_censored"})
    )
    relation["left_to_right"] = relation["left_censored"] / relation["right_censored"].replace(0, np.nan)
    relation["weighted_left_likelihood"] = relation["left_censored"] * 0.25
    relation["weighted_right_likelihood"] = relation["right_censored"] * 0.10
    relation["weighted_left_to_right"] = (
        relation["weighted_left_likelihood"] / relation["weighted_right_likelihood"].replace(0, np.nan)
    )
    write_csv(relation, output / "r2_5_relation_counts_by_target.csv")

    split_rows = []
    for target, target_frame in dataset.groupby("target_chembl_id"):
        for split in SPLITS:
            column = f"split_{split.lower()}"
            counts = target_frame[column].value_counts()
            train = target_frame.loc[target_frame[column].eq("train")]
            test = target_frame.loc[target_frame[column].eq("test")]
            row = {
                "target": target,
                "split": split,
                "train_rows": int(counts.get("train", 0)),
                "val_rows": int(counts.get("val", 0)),
                "test_rows": int(counts.get("test", 0)),
                "test_exact": int(test["p_relation"].eq("=").sum()),
                "test_left": int(test["p_relation"].eq("<").sum()),
                "test_right": int(test["p_relation"].eq(">").sum()),
                "train_groups": int(train["leakage_group"].nunique()),
                "test_groups": int(test["leakage_group"].nunique()),
            }
            for field in ("parent_inchikey", "scaffold", "assay_chembl_id", "document_chembl_id", "leakage_group"):
                row[f"train_test_{field}_overlap"] = int(
                    len(set(train[field].dropna()) & set(test[field].dropna()))
                )
            split_rows.append(row)
    split_audit = pd.DataFrame(split_rows)
    write_csv(split_audit, output / "r2_4_r2_11_split_counts_and_overlap.csv")

    s1_s4 = dataset[["target_chembl_id", "leakage_group", "split_s1", "split_s4"]].copy()
    agreement_by_target = (
        s1_s4.assign(agree=s1_s4["split_s1"].eq(s1_s4["split_s4"]))
        .groupby("target_chembl_id", as_index=False)
        .agg(rows=("agree", "size"), agreeing_rows=("agree", "sum"), agreement=("agree", "mean"))
    )
    write_csv(agreement_by_target, output / "r2_4_s1_s4_row_agreement_by_target.csv")
    cross = pd.crosstab(dataset["split_s1"], dataset["split_s4"]).rename_axis("S1").reset_index()
    write_csv(cross, output / "r2_4_s1_s4_partition_crosstab.csv")

    component = (
        s1_s4.groupby(["target_chembl_id", "leakage_group"], as_index=False)
        .agg(
            rows=("leakage_group", "size"),
            s1_assignments=("split_s1", "nunique"),
            s4_assignments=("split_s4", "nunique"),
            split_s1=("split_s1", "first"),
            split_s4=("split_s4", "first"),
        )
    )
    write_csv(component, output / "r2_4_s1_s4_component_assignments.csv")

    decade = dataset.copy()
    decade["decade"] = np.where(
        decade["document_year"].isna(),
        "Missing",
        (np.floor(decade["document_year"] / 10) * 10).astype("Int64").astype(str) + "-" +
        (np.floor(decade["document_year"] / 10) * 10 + 9).astype("Int64").astype(str),
    )
    decade_table = decade.groupby(["decade", "p_relation"], dropna=False).size().unstack(fill_value=0)
    decade_table = decade_table.reindex(columns=["=", "<", ">"], fill_value=0).reset_index()
    decade_table = decade_table.rename(columns={"=": "exact", "<": "left", ">": "right"})
    decade_table["total"] = decade_table[["exact", "left", "right"]].sum(axis=1)
    for column in ("exact", "left", "right"):
        decade_table[f"{column}_fraction"] = decade_table[column] / decade_table["total"]
    write_csv(decade_table, output / "r2_12_relation_by_decade.csv")

    censored = dataset.loc[dataset["p_relation"].isin(["<", ">"])].copy()
    log_value = np.log10(censored["standard_value"].to_numpy(dtype=float))
    censored["integer_power_of_10"] = np.isclose(log_value, np.round(log_value), atol=1e-10)
    power_summary = (
        censored.groupby("p_relation", as_index=False)
        .agg(rows=("standard_value", "size"), integer_power_rows=("integer_power_of_10", "sum"))
    )
    power_summary["integer_power_fraction"] = power_summary["integer_power_rows"] / power_summary["rows"]
    write_csv(power_summary, output / "r2_2_integer_power_threshold_summary.csv")
    threshold_frequency = (
        censored.groupby(["p_relation", "standard_value", "standard_units"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["p_relation", "rows"], ascending=[True, False])
    )
    write_csv(threshold_frequency, output / "r2_2_threshold_frequency.csv")

    validity = (
        dataset.assign(
            data_validity_comment=dataset["data_validity_comment"].fillna("Missing"),
            activity_comment=dataset["activity_comment"].fillna("Missing"),
        )
        .groupby(["data_validity_comment", "activity_comment"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("rows", ascending=False)
    )
    write_csv(validity, output / "r2_9_quality_comment_cross_counts.csv")

    summary = {
        "rows": int(len(dataset)),
        "targets": int(dataset["target_chembl_id"].nunique()),
        "relations": dataset["p_relation"].value_counts().to_dict(),
        "left_to_right": float(dataset["p_relation"].eq("<").sum() / dataset["p_relation"].eq(">").sum()),
        "weighted_left_to_right": float(
            dataset["p_relation"].eq("<").sum() * 0.25 / (dataset["p_relation"].eq(">").sum() * 0.10)
        ),
        "s1_s4_row_agreement": float(dataset["split_s1"].eq(dataset["split_s4"]).mean()),
        "s1_s4_disagree_rows": int(dataset["split_s1"].ne(dataset["split_s4"]).sum()),
        "s1_s4_component_count": int(len(component)),
        "component_split_violations_s1": int(component["s1_assignments"].gt(1).sum()),
        "component_split_violations_s4": int(component["s4_assignments"].gt(1).sum()),
    }
    return summary


def raw_relation_and_concentration_audit(
    dataset: pd.DataFrame, db_path: Path, output: Path
) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.execute("CREATE TEMP TABLE wanted_targets(chembl_id TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO wanted_targets VALUES (?)",
        [(value,) for value in sorted(dataset["target_chembl_id"].unique())],
    )
    relation_query = """
        SELECT COALESCE(activities.standard_relation, 'NULL') AS standard_relation,
               COUNT(*) AS rows
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN target_dictionary td ON assays.tid = td.tid
        JOIN wanted_targets wt ON wt.chembl_id = td.chembl_id
        JOIN compound_structures cs ON activities.molregno = cs.molregno
        WHERE td.organism = 'Homo sapiens'
          AND td.target_type = 'SINGLE PROTEIN'
          AND activities.standard_type = 'IC50'
          AND activities.standard_units = 'nM'
          AND activities.standard_value IS NOT NULL
          AND activities.standard_value > 0
          AND cs.canonical_smiles IS NOT NULL
        GROUP BY COALESCE(activities.standard_relation, 'NULL')
        ORDER BY rows DESC
    """
    raw_relations = pd.read_sql_query(relation_query, connection)
    raw_relations["fraction_of_basic_filter"] = raw_relations["rows"] / raw_relations["rows"].sum()
    write_csv(raw_relations, output / "r2_12_raw_relation_states.csv")

    relation_detail = pd.read_sql_query(
        """
        SELECT COALESCE(activities.standard_relation, 'NULL') AS standard_relation,
               activities.standard_value
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN target_dictionary td ON assays.tid = td.tid
        JOIN wanted_targets wt ON wt.chembl_id = td.chembl_id
        JOIN compound_structures cs ON activities.molregno = cs.molregno
        WHERE td.organism = 'Homo sapiens'
          AND td.target_type = 'SINGLE PROTEIN'
          AND activities.standard_type = 'IC50'
          AND activities.standard_units = 'nM'
          AND activities.standard_value IS NOT NULL
          AND activities.standard_value > 0
          AND cs.canonical_smiles IS NOT NULL
        """,
        connection,
    )
    relation_detail["p_activity_from_nM"] = 9.0 - np.log10(
        relation_detail["standard_value"].to_numpy(dtype=float)
    )
    distribution = (
        relation_detail.groupby("standard_relation")["p_activity_from_nM"]
        .agg(
            rows="size",
            mean="mean",
            std="std",
            minimum="min",
            q25=lambda values: values.quantile(0.25),
            median="median",
            q75=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    write_csv(distribution, output / "r2_12_raw_relation_activity_distribution.csv")

    processed_relations = (
        dataset["standard_relation"].fillna("NULL").value_counts().rename_axis("standard_relation")
        .rename("processed_rows").reset_index()
    )
    relation_flow = raw_relations[["standard_relation", "rows"]].rename(
        columns={"rows": "basic_filter_rows"}
    ).merge(processed_relations, on="standard_relation", how="outer").fillna(0)
    relation_flow["excluded_rows"] = (
        relation_flow["basic_filter_rows"] - relation_flow["processed_rows"]
    )
    write_csv(relation_flow, output / "r2_12_relation_filter_flow.csv")

    raw_count = dict(zip(raw_relations["standard_relation"], raw_relations["rows"]))
    recognized_censored = sum(raw_count.get(value, 0) for value in ["<", "<=", ">", ">="])
    exact_like = raw_count.get("=", 0) + raw_count.get("~", 0)
    strict_censored = raw_count.get("<", 0) + raw_count.get(">", 0)
    denominator_rows = [
        {
            "definition": "final_processed_panel",
            "censored_rows": int(dataset["p_relation"].ne("=").sum()),
            "denominator_rows": int(len(dataset)),
        },
        {
            "definition": "basic_filter_all_relation_states_unknown_not_censored",
            "censored_rows": int(recognized_censored),
            "denominator_rows": int(raw_relations["rows"].sum()),
        },
        {
            "definition": "recognized_exact_like_or_inequality_only",
            "censored_rows": int(recognized_censored),
            "denominator_rows": int(exact_like + recognized_censored),
        },
        {
            "definition": "strict_equal_less_greater_only",
            "censored_rows": int(strict_censored),
            "denominator_rows": int(raw_count.get("=", 0) + strict_censored),
        },
    ]
    denominator = pd.DataFrame(denominator_rows)
    denominator["censored_fraction"] = denominator["censored_rows"] / denominator["denominator_rows"]
    write_csv(denominator, output / "r2_12_censoring_fraction_denominator_sensitivity.csv")

    validity_catalog = pd.read_sql_query(
        """
        SELECT COALESCE(data_validity_comment, 'Missing') AS data_validity_comment,
               COUNT(*) AS rows
        FROM activities
        GROUP BY COALESCE(data_validity_comment, 'Missing')
        ORDER BY rows DESC
        """,
        connection,
    )
    validity_panel = (
        dataset["data_validity_comment"].fillna("Missing").value_counts()
        .rename_axis("data_validity_comment").rename("rows").reset_index()
    )
    write_csv(validity_catalog, output / "r2_9_chembl35_validity_comment_catalog.csv")
    write_csv(validity_panel, output / "r2_9_ic50_panel_validity_comment_counts.csv")

    censored = dataset.loc[dataset["p_relation"].isin(["<", ">"])].copy()
    assay_ids = sorted(censored["assay_chembl_id"].dropna().unique())
    connection.execute("CREATE TEMP TABLE wanted_assays(chembl_id TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO wanted_assays VALUES (?)", [(value,) for value in assay_ids])
    descriptions = pd.read_sql_query(
        """
        SELECT assays.chembl_id AS assay_chembl_id, assays.description
        FROM assays JOIN wanted_assays wa ON wa.chembl_id = assays.chembl_id
        """,
        connection,
    )
    parameters = pd.read_sql_query(
        """
        SELECT assays.chembl_id AS assay_chembl_id, ap.type, ap.standard_type,
               ap.standard_value, ap.standard_units, ap.standard_text_value
        FROM assay_parameters ap
        JOIN assays ON assays.assay_id = ap.assay_id
        JOIN wanted_assays wa ON wa.chembl_id = assays.chembl_id
        WHERE lower(COALESCE(ap.type, '')) LIKE '%concentration%'
           OR lower(COALESCE(ap.standard_type, '')) LIKE '%concentration%'
        """,
        connection,
    )
    write_csv(parameters, output / "r2_2_structured_assay_concentration_parameters.csv")

    unit_factor = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6}
    patterns = [
        re.compile(
            r"(?:maximum|maximal|highest)(?:\s+(?:tested|test))?\s+concentration(?:\s+of)?\s*[:=]?\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*(pM|nM|uM|um|mM)",
            re.I,
        ),
        re.compile(r"(?:up\s+to)\s*([0-9]+(?:\.[0-9]+)?)\s*(pM|nM|uM|um|mM)", re.I),
        re.compile(
            r"concentrations?\s+(?:ranging\s+)?from\s+[0-9]+(?:\.[0-9]+)?\s*(?:pM|nM|uM|um|mM)?\s+to\s+"
            r"([0-9]+(?:\.[0-9]+)?)\s*(pM|nM|uM|um|mM)",
            re.I,
        ),
    ]
    extracted = []
    for row in descriptions.itertuples(index=False):
        text = str(row.description or "").replace("µ", "u").replace("μ", "u")
        values = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = float(match.group(1)) * unit_factor[match.group(2).lower()]
                values.append((value, match.group(0)))
        if values:
            value, evidence = max(values, key=lambda item: item[0])
            extracted.append(
                {
                    "assay_chembl_id": row.assay_chembl_id,
                    "explicit_max_nM": value,
                    "evidence_text": evidence,
                    "assay_description": text,
                }
            )
    extracted_frame = pd.DataFrame(extracted)
    write_csv(extracted_frame, output / "r2_2_explicit_max_concentration_from_descriptions.csv")
    if len(extracted_frame):
        joined = censored.merge(extracted_frame, on="assay_chembl_id", how="left")
        joined["threshold_equals_explicit_max"] = np.isclose(
            joined["standard_value"], joined["explicit_max_nM"], rtol=1e-6, atol=1e-9, equal_nan=False
        )
    else:
        joined = censored.copy()
        joined["explicit_max_nM"] = np.nan
        joined["threshold_equals_explicit_max"] = False
    write_csv(
        joined.loc[joined["explicit_max_nM"].notna()],
        output / "r2_2_threshold_vs_explicit_max_rows.csv",
    )
    covered = joined["explicit_max_nM"].notna()
    return {
        "raw_relation_states": dict(zip(raw_relations["standard_relation"], raw_relations["rows"])),
        "relation_rows_excluded_after_basic_filter": int(relation_flow["excluded_rows"].sum()),
        "chembl35_validity_states_including_missing": int(len(validity_catalog)),
        "chembl35_nonmissing_validity_comment_types": int(
            validity_catalog["data_validity_comment"].ne("Missing").sum()
        ),
        "ic50_panel_nonmissing_validity_comment_types": int(
            validity_panel["data_validity_comment"].ne("Missing").sum()
        ),
        "censored_assays": len(assay_ids),
        "assays_with_structured_concentration_parameter": int(parameters["assay_chembl_id"].nunique()),
        "assays_with_explicit_max_in_description": int(extracted_frame["assay_chembl_id"].nunique()) if len(extracted_frame) else 0,
        "censored_rows_with_explicit_max": int(covered.sum()),
        "matching_rows_among_explicit": int(joined.loc[covered, "threshold_equals_explicit_max"].sum()),
        "matching_fraction_among_explicit": float(joined.loc[covered, "threshold_equals_explicit_max"].mean()) if covered.any() else np.nan,
        "interpretation": "Description extraction is restricted to explicit maximum/highest/up-to/range language; unresolved assays remain missing.",
    }


def prediction_reanalyses(
    predictions: pd.DataFrame, dataset: pd.DataFrame, output: Path
) -> dict[str, object]:
    test = predictions.loc[predictions["partition"].eq("test")].copy()
    keys = ["treatment", "target_chembl_id", "split", "seed"]
    rows = []
    for key, group in test.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(cell_metrics(group))
        rows.append(row)
    cells = pd.DataFrame(rows)
    cells["method"] = cells["treatment"].map(TREATMENT_LABELS)
    write_csv(cells, output / "r2_8_r2_10d_r2_11_test_cell_metrics.csv")

    metric_columns = [
        "exact_mae",
        "exact_rmse",
        "exact_r2",
        "exact_spearman",
        "exact_kendall",
        "censored_nll_mean",
        "censored_nll_median",
        "cdir",
        "violation_distance",
    ]
    target_metrics = cells.groupby(["treatment", "target_chembl_id"], as_index=False)[metric_columns].mean()
    write_csv(target_metrics, output / "r2_8_r2_10d_target_metrics.csv")
    summary = target_metrics.groupby("treatment")[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns.to_flat_index()]
    summary["method"] = summary["treatment"].map(TREATMENT_LABELS)
    write_csv(summary, output / "r2_8_r2_10d_target_balanced_summary.csv")

    paired = target_metrics.pivot(index="target_chembl_id", columns="treatment", values=metric_columns)
    effect_rows = []
    for metric in metric_columns:
        if (metric, "censored_likelihood") not in paired or (metric, "censored_as_exact") not in paired:
            continue
        delta = paired[(metric, "censored_likelihood")] - paired[(metric, "censored_as_exact")]
        favorable = delta <= 0 if metric in {"exact_mae", "exact_rmse", "censored_nll_mean", "censored_nll_median", "violation_distance"} else delta >= 0
        effect_rows.append(
            {
                "metric": metric,
                "canal_minus_matched": float(delta.mean()),
                "target_sd": float(delta.std(ddof=1)),
                "favorable_targets": int(favorable.sum()),
                "targets": int(delta.notna().sum()),
            }
        )
    effects = pd.DataFrame(effect_rows)
    write_csv(effects, output / "r2_8_r2_10d_canal_vs_matched_effects.csv")
    write_csv(cells.loc[cells["split"].eq("S3")], output / "r2_11_s3_cell_metrics.csv")

    sigma_rows = []
    for target, target_frame in dataset.groupby("target_chembl_id"):
        for split in SPLITS:
            train = target_frame.loc[target_frame[f"split_{split.lower()}"].eq("train")]
            exact = train.loc[train["p_relation"].eq("="), "p_activity"].to_numpy(dtype=float)
            sigma_rows.append(
                {
                    "target_chembl_id": target,
                    "split": split,
                    "train_exact_n": len(exact),
                    "sigma_common": max(float(np.std(exact, ddof=0)), 0.05),
                }
            )
    sigma_frame = pd.DataFrame(sigma_rows)
    write_csv(sigma_frame, output / "r2_10c_common_sigma_by_target_split.csv")
    common = test.merge(sigma_frame, on=["target_chembl_id", "split"], validate="many_to_one")
    common = common.loc[common["p_relation"].ne("=")].copy()
    common["common_sigma_nll"] = stable_censored_nll(
        common["p_activity"].to_numpy(),
        common["p_relation"].to_numpy(),
        common["pred"].to_numpy(),
        common["sigma_common"].to_numpy(),
    )
    common_cells = (
        common.groupby(keys, as_index=False)
        .agg(common_sigma_nll=("common_sigma_nll", "mean"), censored_rows=("common_sigma_nll", "size"))
    )
    write_csv(common_cells, output / "r2_10c_common_sigma_cell_metrics.csv")
    common_targets = common_cells.groupby(["treatment", "target_chembl_id"], as_index=False)["common_sigma_nll"].mean()
    common_summary = common_targets.groupby("treatment", as_index=False).agg(
        mean=("common_sigma_nll", "mean"), sd=("common_sigma_nll", "std")
    )
    write_csv(common_summary, output / "r2_10c_common_sigma_target_balanced_summary.csv")

    baseline_sigma = (
        test.groupby(["treatment", "target_chembl_id", "split", "seed"], as_index=False)["sigma"].first()
    )
    write_csv(baseline_sigma, output / "r2_10c_model_sigma_distribution.csv")

    spearman_effect = effects.loc[effects["metric"].eq("exact_spearman")].iloc[0].to_dict()
    common_map = dict(zip(common_summary["treatment"], common_summary["mean"]))
    return {
        "prediction_rows": int(len(predictions)),
        "test_rows": int(len(test)),
        "cells": int(len(cells)),
        "cell_coverage": cells.groupby("treatment").size().to_dict(),
        "spearman_canal_minus_matched": float(spearman_effect["canal_minus_matched"]),
        "spearman_favorable_targets": int(spearman_effect["favorable_targets"]),
        "common_sigma_nll_matched": float(common_map.get("censored_as_exact", np.nan)),
        "common_sigma_nll_canal": float(common_map.get("censored_likelihood", np.nan)),
        "common_sigma_nll_gain": float(
            common_map.get("censored_as_exact", np.nan) - common_map.get("censored_likelihood", np.nan)
        ),
    }


def candidate_statistics(frame: pd.DataFrame, slopes: np.ndarray, offsets: np.ndarray) -> pd.DataFrame:
    truth = frame["p_activity"].to_numpy(dtype=float)
    base = frame["pred"].to_numpy(dtype=float)
    relation = frame["p_relation"].astype(str).to_numpy()
    exact = relation == "="
    censored = ~exact
    rows = []
    for start in range(0, len(slopes), 128):
        a = slopes[start : start + 128]
        b = offsets[start : start + 128]
        pred = base[:, None] * a[None, :] + b[None, :]
        mae = np.mean(np.abs(pred[exact] - truth[exact, None]), axis=0)
        if censored.sum():
            compatible = np.where(
                relation[censored, None] == "<",
                pred[censored] < truth[censored, None],
                pred[censored] > truth[censored, None],
            )
            cdir = compatible.mean(axis=0)
            violation = np.where(
                relation[censored, None] == "<",
                np.maximum(pred[censored] - truth[censored, None], 0),
                np.maximum(truth[censored, None] - pred[censored], 0),
            ).mean(axis=0)
        else:
            cdir = np.full(len(a), np.nan)
            violation = np.full(len(a), np.nan)
        rows.append(pd.DataFrame({"a": a, "b": b, "mae": mae, "cdir": cdir, "violation": violation}))
    return pd.concat(rows, ignore_index=True)


def affine_reanalysis(predictions: pd.DataFrame, output: Path) -> dict[str, object]:
    if not {"val", "test"}.issubset(set(predictions["partition"].unique())):
        return {"status": "not_run", "reason": "validation predictions unavailable"}
    baseline = predictions.loc[predictions["treatment"].eq("censored_as_exact")].copy()
    canal_test = predictions.loc[
        predictions["treatment"].eq("censored_likelihood") & predictions["partition"].eq("test")
    ].copy()
    broad_a = np.round(np.arange(0.10, 3.0001, 0.10), 10)
    broad_b = np.round(np.arange(-5.00, 5.0001, 0.10), 10)
    fine_a = np.round(np.arange(0.50, 1.5001, 0.05), 10)
    fine_b = np.round(np.arange(-1.50, 1.5001, 0.05), 10)
    broad_slopes, broad_offsets = np.meshgrid(broad_a, broad_b, indexing="ij")
    fine_slopes, fine_offsets = np.meshgrid(fine_a, fine_b, indexing="ij")
    candidates = np.unique(
        np.vstack(
            [
                np.column_stack([broad_slopes.ravel(), broad_offsets.ravel()]),
                np.column_stack([fine_slopes.ravel(), fine_offsets.ravel()]),
            ]
        ),
        axis=0,
    )
    slopes = candidates[:, 0]
    offsets = candidates[:, 1]
    tolerances = [0.00, 0.01, 0.02, 0.04, 0.08, 0.12]
    keys = ["target_chembl_id", "split", "seed"]
    rows = []
    for key, group in baseline.groupby(keys, sort=True):
        val = group.loc[group["partition"].eq("val")]
        test = group.loc[group["partition"].eq("test")]
        stats = candidate_statistics(val, slopes, offsets)
        identity_mae = float(stats.loc[np.isclose(stats["a"], 1) & np.isclose(stats["b"], 0), "mae"].iloc[0])
        for tolerance in tolerances:
            feasible = stats.loc[stats["mae"].le(identity_mae + tolerance + 1e-12)].copy()
            feasible["deviation"] = np.abs(feasible["a"] - 1.0) + np.abs(feasible["b"])
            for strategy in ("maximize_cdir", "minimize_violation"):
                if strategy == "maximize_cdir":
                    chosen = feasible.sort_values(
                        ["cdir", "mae", "deviation", "a", "b"],
                        ascending=[False, True, True, True, True],
                    ).iloc[0]
                else:
                    chosen = feasible.sort_values(
                        ["violation", "mae", "deviation", "a", "b"],
                        ascending=[True, True, True, True, True],
                    ).iloc[0]
                transformed = test["pred"].to_numpy(dtype=float) * chosen["a"] + chosen["b"]
                row = dict(zip(keys, key))
                row.update(
                    {
                        "strategy": strategy,
                        "mae_tolerance": tolerance,
                        "a": float(chosen["a"]),
                        "b": float(chosen["b"]),
                        "validation_identity_mae": identity_mae,
                        "validation_selected_mae": float(chosen["mae"]),
                        "validation_selected_cdir": float(chosen["cdir"]),
                        "validation_selected_violation": float(chosen["violation"]),
                    }
                )
                row.update({f"test_{name}": value for name, value in cell_metrics(test, transformed).items()})
                rows.append(row)
    cells = pd.DataFrame(rows)
    write_csv(cells, output / "r2_1_affine_selected_cell_metrics.csv")
    metrics = ["test_exact_mae", "test_cdir", "test_violation_distance", "test_censored_nll_mean"]
    target = cells.groupby(["strategy", "mae_tolerance", "target_chembl_id"], as_index=False)[metrics].mean()
    write_csv(target, output / "r2_1_affine_target_metrics.csv")
    summary = target.groupby(["strategy", "mae_tolerance"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns.to_flat_index()]
    write_csv(summary, output / "r2_1_affine_target_balanced_summary.csv")

    canal_rows = []
    for key, group in canal_test.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(cell_metrics(group))
        canal_rows.append(row)
    canal_cells = pd.DataFrame(canal_rows)
    canal_target = canal_cells.groupby("target_chembl_id", as_index=False).mean(numeric_only=True)
    canal = {column: float(canal_target[column].mean()) for column in ["exact_mae", "cdir", "violation_distance", "censored_nll_mean"]}
    reachability = summary.copy()
    reachability["canal_exact_mae"] = canal["exact_mae"]
    reachability["canal_cdir"] = canal["cdir"]
    reachability["canal_violation_distance"] = canal["violation_distance"]
    reachability["same_or_lower_mae_than_canal"] = reachability["test_exact_mae_mean"].le(canal["exact_mae"])
    reachability["reaches_canal_cdir"] = reachability["test_cdir_mean"].ge(canal["cdir"])
    reachability["reaches_canal_violation"] = reachability["test_violation_distance_mean"].le(
        canal["violation_distance"]
    )
    write_csv(reachability, output / "r2_1_affine_reachability.csv")
    return {
        "status": "complete",
        "grid": {
            "broad_a": [0.1, 3.0, 0.1],
            "broad_b": [-5.0, 5.0, 0.1],
            "fine_a": [0.5, 1.5, 0.05],
            "fine_b": [-1.5, 1.5, 0.05],
            "unique_candidates": int(len(candidates)),
        },
        "cells": int(len(cells)),
        "canal": canal,
        "any_cdir_reachable_at_same_or_lower_mae": bool(
            (reachability["same_or_lower_mae_than_canal"] & reachability["reaches_canal_cdir"]).any()
        ),
        "any_violation_reachable_at_same_or_lower_mae": bool(
            (reachability["same_or_lower_mae_than_canal"] & reachability["reaches_canal_violation"]).any()
        ),
    }


def s5_correction(root: Path, output: Path) -> dict[str, object]:
    path = root / "reports/evidence_completion_2026-09-03/source_context_s5_results_runs.csv"
    runs = pd.read_csv(path)
    relevant = runs.loc[
        runs["status"].eq("ok")
        & runs["metadata_mode"].eq("source_light")
        & runs["treatment"].isin(["censored_as_exact", "censored_likelihood"])
    ].copy()
    metrics = ["test_censored_nll", "test_censored_dir_acc"]
    means = relevant.groupby(["target", "treatment"])[metrics].mean().unstack("treatment")
    corrected = pd.DataFrame(index=means.index)
    corrected["s5_nll_gain"] = means[("test_censored_nll", "censored_as_exact")] - means[
        ("test_censored_nll", "censored_likelihood")
    ]
    corrected["s5_cdir_gain"] = means[("test_censored_dir_acc", "censored_likelihood")] - means[
        ("test_censored_dir_acc", "censored_as_exact")
    ]
    corrected = corrected.reset_index()
    write_csv(corrected, output / "r2_3_s5_corrected_target_gains.csv")
    return {
        "nll_gain": float(corrected["s5_nll_gain"].mean()),
        "nll_wins": int(corrected["s5_nll_gain"].gt(0).sum()),
        "cdir_gain": float(corrected["s5_cdir_gain"].mean()),
        "cdir_wins": int(corrected["s5_cdir_gain"].gt(0).sum()),
        "targets": int(len(corrected)),
        "bug": "The previous S17 aggregation grouped by target and treatment without filtering metadata_mode, averaging source_light CANAL with the metadata_mode=none ablation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--chembl-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_path = (root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset
    db_path = (root / args.chembl_db).resolve() if not args.chembl_db.is_absolute() else args.chembl_db
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(dataset_path).reset_index(drop=True)
    dataset["row_id"] = dataset.index
    patterns = [str((root / value).resolve()) if not Path(value).is_absolute() else value for value in args.predictions]
    predictions, prediction_files = load_predictions(patterns)

    report = {
        "dataset_audit": audit_relations_and_splits(dataset, output),
        "raw_data_audit": raw_relation_and_concentration_audit(dataset, db_path, output),
        "prediction_reanalyses": prediction_reanalyses(predictions, dataset, output),
        "affine_reanalysis": affine_reanalysis(predictions, output),
        "s5_correction": s5_correction(root, output),
        "provenance": {
            "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path)},
            "chembl_db": {"path": str(db_path), "sha256": sha256(db_path)},
            "predictions": [
                {"path": str(path), "sha256": sha256(path)} for path in prediction_files
            ],
        },
    }
    (output / "reviewer2_validation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
