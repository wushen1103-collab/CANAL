#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def mol_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def train_fps_by_target_split(dataset: pd.DataFrame, splits: list[str]) -> dict[tuple[str, str], list]:
    cache: dict[tuple[str, str], list] = {}
    for split in splits:
        split_col = f"split_{split.lower()}"
        if split_col not in dataset:
            continue
        train = dataset[dataset[split_col].eq("train")]
        for target, group in train.groupby("target_chembl_id"):
            fps = [fp for fp in group["canonical_smiles_rdkit"].dropna().map(mol_fp).tolist() if fp is not None]
            cache[(target, split)] = fps
    return cache


def max_tanimoto(smiles: str, fps: list) -> float:
    fp = mol_fp(smiles)
    if fp is None or not fps:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return float(max(sims)) if sims else np.nan


def attach_dataset_columns(pred: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "canonical_smiles_rdkit",
        "target_pref_name",
        "standard_relation",
        "standard_value",
        "standard_units",
    ]
    src = dataset.reset_index().rename(columns={"index": "row_id"})
    return pred.merge(src[["row_id", *[c for c in cols if c in src.columns]]], on="row_id", how="left")


def make_wide(pred: pd.DataFrame) -> pd.DataFrame:
    cens = pred[~pred["is_exact"].fillna(False).astype(bool)].copy()
    index_cols = [
        "row_id",
        "molecule_chembl_id",
        "parent_inchikey",
        "canonical_smiles_rdkit",
        "target_chembl_id",
        "target_pref_name",
        "assay_chembl_id",
        "document_chembl_id",
        "split",
        "p_relation",
        "p_activity",
        "standard_relation",
        "standard_value",
        "standard_units",
        "scaffold",
    ]
    index_cols = [c for c in index_cols if c in cens.columns]
    wide = cens.pivot_table(
        index=index_cols,
        columns="treatment",
        values=["pred", "censored_nll", "censored_dir_ok"],
        aggfunc="first",
    )
    wide.columns = ["_".join(str(x) for x in c if str(x)) for c in wide.columns]
    return wide.reset_index()


def summarize_cases(wide: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    required = [
        "pred_exact_only",
        "pred_censored_likelihood",
        "censored_nll_exact_only",
        "censored_nll_censored_likelihood",
        "censored_dir_ok_exact_only",
        "censored_dir_ok_censored_likelihood",
    ]
    missing = [c for c in required if c not in wide]
    if missing:
        raise ValueError(f"Missing prediction treatments/columns: {missing}")
    wide = wide.copy()
    wide["nll_gain_vs_exact"] = wide["censored_nll_exact_only"] - wide["censored_nll_censored_likelihood"]
    wide["canal_fixes_exact_violation"] = (
        wide["censored_dir_ok_exact_only"].astype("boolean").eq(False)
        & wide["censored_dir_ok_censored_likelihood"].astype("boolean").eq(True)
    )
    fixed = wide[wide["canal_fixes_exact_violation"]].sort_values("nll_gain_vs_exact", ascending=False)
    selected = []
    for _, group in fixed.groupby("target_chembl_id", sort=False):
        selected.append(group.head(2))
        if len(selected) >= 3:
            break
    cases = pd.concat(selected, ignore_index=True) if selected else fixed.head(max_cases)
    return cases.head(max_cases)


def exact_neighbor_context(pred: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exact = pred[pred["is_exact"].fillna(False).astype(bool)].copy()
    for _, case in cases.iterrows():
        same_parent = exact[
            exact["parent_inchikey"].eq(case["parent_inchikey"])
            & exact["target_chembl_id"].eq(case["target_chembl_id"])
        ]
        same_scaffold = exact[
            exact["scaffold"].eq(case.get("scaffold", ""))
            & exact["target_chembl_id"].eq(case["target_chembl_id"])
        ]
        context = same_parent if not same_parent.empty else same_scaffold
        row = {
            "case_row_id": case["row_id"],
            "target_chembl_id": case["target_chembl_id"],
            "context_type": "same_parent" if not same_parent.empty else "same_scaffold",
            "neighbor_exact_rows": int(context["row_id"].nunique()) if not context.empty else 0,
        }
        if not context.empty:
            summary = context.groupby("treatment")["abs_error"].mean().to_dict()
            row["neighbor_exact_abs_error_exact_only"] = summary.get("exact_only", np.nan)
            row["neighbor_exact_abs_error_canal"] = summary.get("censored_likelihood", np.nan)
            row["neighbor_exact_pactivity_min"] = float(context["p_activity"].min())
            row["neighbor_exact_pactivity_max"] = float(context["p_activity"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def similarity_strata(pred: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.3, 0.5, 0.7, np.inf]
    labels = ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]
    pred = pred.copy()
    pred["similarity_bin"] = pd.cut(pd.to_numeric(pred["max_train_tanimoto"], errors="coerce"), bins=bins, labels=labels)
    rows = []
    for (sim_bin, relation, treatment), group in pred.groupby(["similarity_bin", "p_relation", "treatment"], dropna=False):
        exact = group["is_exact"].fillna(False).astype(bool)
        cens = ~exact
        rows.append(
            {
                "similarity_bin": str(sim_bin),
                "relation": relation,
                "treatment": treatment,
                "rows": len(group),
                "mae": pd.to_numeric(group.loc[exact, "abs_error"], errors="coerce").mean(),
                "censored_nll": pd.to_numeric(group.loc[cens, "censored_nll"], errors="coerce").mean(),
                "censored_dir": pd.to_numeric(group.loc[cens, "censored_dir_ok"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=Path("reports/ic50_chemical_case_studies_2026-08-12"))
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_parquet(args.predictions)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    pred = attach_dataset_columns(pred, dataset)
    splits = sorted(pred["split"].dropna().astype(str).unique())
    fp_cache = train_fps_by_target_split(dataset, splits)
    unique_rows = pred.drop_duplicates(["row_id", "split"])[
        ["row_id", "canonical_smiles_rdkit", "target_chembl_id", "split"]
    ].copy()
    unique_rows["max_train_tanimoto"] = [
        max_tanimoto(row.canonical_smiles_rdkit, fp_cache.get((row.target_chembl_id, row.split), []))
        for row in unique_rows.itertuples(index=False)
    ]
    pred = pred.merge(
        unique_rows[["row_id", "split", "max_train_tanimoto"]],
        on=["row_id", "split"],
        how="left",
        validate="many_to_one",
    )
    wide = make_wide(pred)
    cases = summarize_cases(wide, args.max_cases)
    context = exact_neighbor_context(pred, cases)
    sim = similarity_strata(pred)

    pred.head(5000).to_csv(args.out_prefix.with_suffix(".pred_head.csv"), index=False)
    cases.to_csv(args.out_prefix.with_suffix(".cases.csv"), index=False)
    context.to_csv(args.out_prefix.with_suffix(".context.csv"), index=False)
    sim.to_csv(args.out_prefix.with_suffix(".similarity_strata.csv"), index=False)
    pred[["row_id", "split", "target_chembl_id", "max_train_tanimoto"]].drop_duplicates().to_csv(
        args.out_prefix.with_suffix(".row_similarity.csv"), index=False
    )
    print(f"[done] wrote {args.out_prefix.with_suffix('.cases.csv')}")
    print(f"[done] wrote {args.out_prefix.with_suffix('.similarity_strata.csv')}")


if __name__ == "__main__":
    main()
