#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import zipfile
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

ENDPOINT_COLUMNS = {
    "IC50": "IC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
}

VALUE_RE = re.compile(r"^\s*(<=|>=|<|>|=|~)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_nm_value(text: object) -> tuple[str, float] | None:
    s = str(text or "").strip().replace(",", "")
    if not s:
        return None
    match = VALUE_RE.match(s)
    if not match:
        return None
    rel = match.group(1) or "="
    if rel == "~":
        rel = "="
    val = float(match.group(2))
    if not math.isfinite(val) or val <= 0:
        return None
    return rel, val


def parse_year(*texts: object) -> int | None:
    for text in texts:
        s = str(text or "").strip()
        if not s:
            continue
        match = re.search(r"(19|20)\d{2}", s)
        if match:
            year = int(match.group(0))
            if 1900 <= year <= 2100:
                return year
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                year = datetime.strptime(s, fmt).year
                if 1900 <= year <= 2100:
                    return year
            except ValueError:
                pass
    return None


def first_nonempty(row: dict[str, str], cols: list[str]) -> str:
    for col in cols:
        val = str(row.get(col, "") or "").strip()
        if val:
            return val
    return ""


def canonical_scaffold_inchikey(smiles: str) -> tuple[str | None, str | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    if not scaffold:
        scaffold = can
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = None
    return can, scaffold, inchikey


def normalize_relation(rel: str) -> str:
    if rel in {"<", "<="}:
        return "lower"
    if rel in {">", ">="}:
        return "upper"
    return "exact"


def row_to_records(row: dict[str, str], endpoints: list[str], allow_multichain: bool) -> list[dict[str, object]]:
    chain_count_text = str(row.get("Number of Protein Chains in Target (>1 implies a multichain complex)", "") or "").strip()
    if chain_count_text:
        try:
            chain_count = int(float(chain_count_text))
            if chain_count > 1 and not allow_multichain:
                return []
        except ValueError:
            pass

    smiles = str(row.get("Ligand SMILES", "") or "").strip()
    if not smiles:
        return []
    uniprot = first_nonempty(
        row,
        [
            "UniProt (SwissProt) Primary ID of Target Chain 1",
            "UniProt (TrEMBL) Primary ID of Target Chain 1",
        ],
    )
    sequence = str(row.get("BindingDB Target Chain Sequence 1", "") or "").strip()
    target_name = str(row.get("Target Name", "") or "").strip()
    if not uniprot and not target_name:
        return []
    target_id = uniprot or hashlib.sha1(target_name.encode("utf-8", errors="ignore")).hexdigest()[:12]
    year = parse_year(row.get("Date of publication"), row.get("Date in BindingDB"))
    if year is None:
        return []

    reactant_id = str(row.get("BindingDB Reactant_set_id", "") or "").strip()
    doi = str(row.get("Article DOI", "") or "").strip()
    entry_doi = str(row.get("BindingDB Entry DOI", "") or "").strip()
    pmid = str(row.get("PMID", "") or "").strip()
    article_key = doi or pmid or entry_doi or f"{target_id}:{year}"
    article_hash = hashlib.sha1(article_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
    target_org = str(row.get("Target Source Organism According to Curator or DataSource", "") or "").strip()
    curation = str(row.get("Curation/DataSource", "") or "").strip()

    out = []
    for endpoint in endpoints:
        parsed = parse_nm_value(row.get(ENDPOINT_COLUMNS[endpoint], ""))
        if parsed is None:
            continue
        rel, nm = parsed
        p_threshold = 9.0 - math.log10(nm)
        ctype = normalize_relation(rel)
        record = {
            "activity_id": f"BDB::{reactant_id or article_hash}::{endpoint}",
            "molecule_chembl_id": str(row.get("ChEMBL ID of Ligand", "") or "").strip(),
            "canonical_smiles": smiles,
            "standard_inchi_key": str(row.get("Ligand InChI Key", "") or "").strip(),
            "target_chembl_id": f"BDBTARGET::{target_id}",
            "target_name": target_name,
            "uniprot_accession": uniprot,
            "target_sequence": sequence,
            "endpoint": endpoint,
            "standard_value": nm,
            "standard_value_nm": nm,
            "standard_units": "nM",
            "standard_relation": rel,
            "pchembl_value": p_threshold if rel == "=" else np.nan,
            "activity_comment": curation,
            "data_validity_comment": "",
            "potential_duplicate": 0,
            "activity_src_id": "BindingDB",
            "assay_id": f"BDBARTICLE::{article_hash}",
            "assay_chembl_id": f"BDBARTICLE::{article_hash}",
            "assay_type": "B",
            "assay_test_type": "BindingDB article",
            "assay_category": "binding",
            "assay_organism": target_org,
            "assay_tax_id": "",
            "assay_description": target_name,
            "bao_format": "BindingDB",
            "confidence_score": 8,
            "assay_src_id": "BindingDB",
            "doc_id": f"BDBDOC::{article_hash}",
            "document_chembl_id": f"BDBDOC::{article_hash}",
            "doc_type": "ARTICLE",
            "journal": "",
            "pubmed_id": pmid,
            "doi": doi or entry_doi,
            "publication_year": year,
            "p_activity_threshold": p_threshold,
            "y_observed": p_threshold,
            "is_censored": ctype != "exact",
            "censor_type": ctype,
            "target_endpoint": f"BDBTARGET::{target_id}::{endpoint}",
            "bindingdb_reactant_set_id": reactant_id,
            "bindingdb_article_key": article_key,
            "bindingdb_entry_doi": entry_doi,
            "pubchem_cid": str(row.get("PubChem CID", "") or "").strip(),
            "drugbank_id": str(row.get("DrugBank ID of Ligand", "") or "").strip(),
        }
        out.append(record)
    return out


def select_groups(df: pd.DataFrame, target_cap: int, min_rows: int, min_censor_rate: float, max_censor_rate: float) -> pd.DataFrame:
    stats = (
        df.groupby(["target_endpoint", "target_chembl_id", "target_name", "endpoint"], dropna=False)
        .agg(
            n=("activity_id", "count"),
            exact_n=("censor_type", lambda x: int((x == "exact").sum())),
            censored_n=("is_censored", "sum"),
            molecule_n=("standard_inchi_key", "nunique"),
            source_n=("assay_chembl_id", "nunique"),
            year_min=("publication_year", "min"),
            year_max=("publication_year", "max"),
        )
        .reset_index()
    )
    stats["censor_rate"] = stats["censored_n"] / stats["n"].clip(lower=1)
    cand = stats[
        (stats["n"] >= min_rows)
        & (stats["exact_n"] >= max(50, min_rows // 4))
        & (stats["censored_n"] >= max(10, min_rows // 50))
        & (stats["censor_rate"].between(min_censor_rate, max_censor_rate))
    ].copy()
    if cand.empty:
        cand = stats[(stats["n"] >= min_rows) & (stats["censored_n"] > 0) & (stats["exact_n"] > 0)].copy()
    if cand.empty:
        raise RuntimeError("No BindingDB target-endpoint group has enough exact+censored rows.")
    cand["balance"] = 1.0 - (cand["censor_rate"] - 0.15).abs()
    cand["score"] = np.log1p(cand["n"]) + cand["balance"] + 0.25 * np.log1p(cand["source_n"]) + 0.15 * np.log1p(cand["molecule_n"])
    chosen = []
    used_targets = set()
    for _, row in cand.sort_values("score", ascending=False).iterrows():
        target = str(row["target_chembl_id"])
        if target in used_targets:
            continue
        chosen.append(row["target_endpoint"])
        used_targets.add(target)
        if len(chosen) >= target_cap:
            break
    if len(chosen) < target_cap:
        present = set(chosen)
        for _, row in cand.sort_values("score", ascending=False).iterrows():
            key = row["target_endpoint"]
            if key in present:
                continue
            chosen.append(key)
            present.add(key)
            if len(chosen) >= target_cap:
                break
    return stats[stats["target_endpoint"].isin(chosen)].sort_values("score" if "score" in stats.columns else "n", ascending=False)


def deterministic_downsample(df: pd.DataFrame, target_rows: int | None, max_per_group: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()
    if max_per_group:
        parts = []
        for _, group in out.groupby("target_endpoint", sort=False):
            if len(group) <= max_per_group:
                parts.append(group)
                continue
            sampled = (
                group.groupby("censor_type", group_keys=False)
                .apply(lambda x: x.sample(n=max(1, min(len(x), int(round(max_per_group * len(x) / len(group))))), random_state=seed))
                .sample(frac=1.0, random_state=seed)
                .head(max_per_group)
            )
            parts.append(sampled)
        out = pd.concat(parts, ignore_index=True)
    if target_rows and len(out) > target_rows:
        frac = target_rows / len(out)
        out = (
            out.groupby(["target_endpoint", "censor_type"], group_keys=False)
            .apply(lambda x: x.sample(n=max(1, min(len(x), int(round(len(x) * frac)))), random_state=seed))
            .sample(frac=1.0, random_state=seed)
        )
        if len(out) > target_rows:
            out = out.sample(n=target_rows, random_state=seed)
    return out.reset_index(drop=True)


def add_chemistry(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    unique = pd.Series(df["canonical_smiles"].dropna().unique())
    with Pool(processes=max(1, workers)) as pool:
        rows = list(tqdm(pool.imap(canonical_scaffold_inchikey, unique.tolist(), chunksize=256), total=len(unique), desc="rdkit_bindingdb"))
    cmap = dict(zip(unique.tolist(), rows))
    can, scaffold, computed_inchikey = [], [], []
    for smi in tqdm(df["canonical_smiles"].tolist(), desc="map_bindingdb_chemistry"):
        c, s, ik = cmap.get(smi, (None, None, None))
        can.append(c)
        scaffold.append(s)
        computed_inchikey.append(ik)
    out = df.copy()
    out["canonical_smiles_rdkit"] = can
    out["bemis_murcko_scaffold"] = scaffold
    raw_key = out["standard_inchi_key"].replace("", np.nan)
    out["computed_inchi_key"] = computed_inchikey
    out["molecule_key"] = raw_key.fillna(out["computed_inchi_key"]).fillna(out["canonical_smiles_rdkit"])
    out = out.dropna(subset=["canonical_smiles_rdkit", "molecule_key", "bemis_murcko_scaffold"]).copy()
    out["first_seen_molecule_year"] = out.groupby("molecule_key")["publication_year"].transform("min")
    out["first_seen_scaffold_year"] = out.groupby("bemis_murcko_scaffold")["publication_year"].transform("min")
    out["first_seen_target_year"] = out.groupby("target_chembl_id")["publication_year"].transform("min")
    out["is_first_seen_molecule"] = out["publication_year"].eq(out["first_seen_molecule_year"])
    out["is_first_seen_scaffold"] = out["publication_year"].eq(out["first_seen_scaffold_year"])
    out["is_first_seen_target"] = out["publication_year"].eq(out["first_seen_target_year"])
    return out


def remove_overlap(df: pd.DataFrame, chembl_reference: str | None) -> tuple[pd.DataFrame, int]:
    if not chembl_reference:
        return df, 0
    ref_path = Path(chembl_reference)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    ref = pd.read_parquet(ref_path, columns=["molecule_key", "uniprot_accession", "endpoint"])
    ref_keys = set(map(tuple, ref.dropna(subset=["molecule_key", "uniprot_accession", "endpoint"]).astype(str).to_numpy()))
    this_keys = list(map(tuple, df[["molecule_key", "uniprot_accession", "endpoint"]].fillna("").astype(str).to_numpy()))
    keep = np.array([key not in ref_keys for key in this_keys], dtype=bool)
    removed = int((~keep).sum())
    return df.loc[keep].copy(), removed


def load_bindingdb(zip_path: str | Path, endpoints: list[str], year_min: int, year_max: int, allow_multichain: bool) -> pd.DataFrame:
    records = []
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            import io

            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""), delimiter="\t")
            for row in tqdm(reader, desc="parse_bindingdb"):
                for record in row_to_records(row, endpoints, allow_multichain=allow_multichain):
                    year = int(record["publication_year"])
                    if year_min <= year <= year_max:
                        records.append(record)
    if not records:
        raise RuntimeError("No usable BindingDB records after filtering.")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", default="data/processed/bindingdb_articles_202608_canal20.parquet")
    parser.add_argument("--chembl-reference", default="data/processed/chembl37_canal20.parquet")
    parser.add_argument("--no-remove-overlap", action="store_true")
    parser.add_argument("--endpoints", nargs="+", default=["IC50", "Ki", "Kd", "EC50"], choices=list(ENDPOINT_COLUMNS))
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2026)
    parser.add_argument("--target-cap", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=350)
    parser.add_argument("--min-censor-rate", type=float, default=0.005)
    parser.add_argument("--max-censor-rate", type=float, default=0.90)
    parser.add_argument("--max-per-group", type=int, default=6000)
    parser.add_argument("--target-rows", type=int, default=80000)
    parser.add_argument("--allow-multichain", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=max(1, min(96, (os.cpu_count() or 4) - 30)))
    args = parser.parse_args()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    rows = load_bindingdb(args.zip, args.endpoints, args.year_min, args.year_max, allow_multichain=args.allow_multichain)
    all_stats = (
        rows.groupby(["target_endpoint", "target_chembl_id", "target_name", "endpoint"], dropna=False)
        .agg(
            n=("activity_id", "count"),
            exact_n=("censor_type", lambda x: int((x == "exact").sum())),
            censored_n=("is_censored", "sum"),
            molecule_n=("canonical_smiles", "nunique"),
            source_n=("assay_chembl_id", "nunique"),
            year_min=("publication_year", "min"),
            year_max=("publication_year", "max"),
        )
        .reset_index()
    )
    all_stats["censor_rate"] = all_stats["censored_n"] / all_stats["n"].clip(lower=1)
    all_stats.sort_values("n", ascending=False).to_csv(out_path.with_suffix(".all_group_stats.csv"), index=False)

    selected = select_groups(rows, args.target_cap, args.min_rows, args.min_censor_rate, args.max_censor_rate)
    selected.to_csv(out_path.with_suffix(".selected_groups.csv"), index=False)
    rows = rows[rows["target_endpoint"].isin(set(selected["target_endpoint"]))].copy()
    rows = deterministic_downsample(rows, args.target_rows, args.max_per_group, args.seed)
    rows = add_chemistry(rows, workers=args.workers)

    overlap_removed = 0
    if not args.no_remove_overlap:
        rows, overlap_removed = remove_overlap(rows, args.chembl_reference)
    rows = rows.sort_values(["target_endpoint", "publication_year", "activity_id"]).reset_index(drop=True)
    rows.to_parquet(out_path, index=False)
    manifest = {
        "zip": str(args.zip),
        "zip_sha256": sha256_file(args.zip),
        "out": str(out_path),
        "chembl_reference": None if args.no_remove_overlap else str(args.chembl_reference),
        "overlap_removed_rows": int(overlap_removed),
        "rows": int(len(rows)),
        "target_endpoints": int(rows["target_endpoint"].nunique()),
        "targets": int(rows["target_chembl_id"].nunique()),
        "endpoints": sorted(rows["endpoint"].dropna().unique().tolist()),
        "molecules": int(rows["molecule_key"].nunique()),
        "scaffolds": int(rows["bemis_murcko_scaffold"].nunique()),
        "assays": int(rows["assay_chembl_id"].nunique()),
        "docs": int(rows["document_chembl_id"].nunique()),
        "years": [int(rows["publication_year"].min()), int(rows["publication_year"].max())],
        "censor_rate": float(rows["is_censored"].mean()),
        "relation_counts": rows["standard_relation"].value_counts(dropna=False).to_dict(),
        "censor_type_counts": rows["censor_type"].value_counts(dropna=False).to_dict(),
        "seed": int(args.seed),
        "target_rows_requested": int(args.target_rows) if args.target_rows else None,
        "max_per_group": int(args.max_per_group) if args.max_per_group else None,
        "allow_multichain": bool(args.allow_multichain),
    }
    write_json(out_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
