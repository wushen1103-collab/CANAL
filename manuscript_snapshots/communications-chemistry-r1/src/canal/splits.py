from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SplitReport:
    split: str
    train_rows: int
    val_rows: int
    test_rows: int
    inchikey_train_test_overlap: int
    document_train_test_overlap: int


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def leakage_groups(df: pd.DataFrame) -> pd.Series:
    """Group rows by connected parent InChIKey/document components."""
    return connected_groups(df, ["parent_inchikey", "document_chembl_id"], name="leakage_group")


def connected_groups(df: pd.DataFrame, key_cols: list[str], name: str = "connected_group") -> pd.Series:
    """Group rows by connected components over one or more no-leakage keys."""
    uf = UnionFind()
    row_nodes: list[str] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        row_node = f"row:{idx}"
        row_nodes.append(row_node)
        for col in key_cols:
            value = row.get(col)
            if pd.notna(value) and value:
                uf.union(row_node, f"{col}:{value}")
    roots = [uf.find(node) for node in row_nodes]
    root_to_id = {root: i for i, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_id[root] for root in roots], index=df.index, name=name)


def _assign_groups_by_keys(
    df: pd.DataFrame,
    group_col: str,
    key_col: str,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    group_info = (
        df.groupby(group_col)
        .agg(n=("p_activity", "size"), key=(key_col, lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else "NA"))
        .reset_index()
    )
    key_counts = group_info.groupby("key")["n"].sum().to_dict()
    total = sum(key_counts.values())
    targets = {"train": total * (1.0 - val_frac - test_frac), "val": total * val_frac, "test": total * test_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    split_keys = {"train": set(), "val": set(), "test": set()}

    keys = list(key_counts)
    rng.shuffle(keys)
    keys = sorted(keys, key=lambda k: key_counts[k], reverse=True)
    for key in keys:
        split = max(targets, key=lambda s: targets[s] - counts[s])
        split_keys[split].add(key)
        counts[split] += key_counts.get(key, 0)

    group_to_split = {}
    for _, row in group_info.iterrows():
        for split, keys_for_split in split_keys.items():
            if row["key"] in keys_for_split:
                group_to_split[row[group_col]] = split
                break
    return df[group_col].map(group_to_split)


def _assign_random_groups(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    shuffled = df.groupby(group_col).size().sample(frac=1.0, random_state=seed)
    group_sizes = shuffled.sort_values(ascending=False)
    total = int(group_sizes.sum())
    targets = {"train": total * (1.0 - val_frac - test_frac), "val": total * val_frac, "test": total * test_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    assignment: dict[int, str] = {}
    for group, n in group_sizes.items():
        split = max(targets, key=lambda s: targets[s] - counts[s])
        assignment[group] = split
        counts[split] += int(n)
    return df[group_col].map(assignment)


def _assign_temporal(
    df: pd.DataFrame,
    group_col: str,
    val_frac: float,
    test_frac: float,
) -> pd.Series:
    tmp = df.copy()
    years = pd.to_numeric(tmp["document_year"], errors="coerce")
    if years.notna().sum() < max(100, 0.2 * len(tmp)):
        return _assign_random_groups(tmp, group_col, seed=2081, val_frac=val_frac, test_frac=test_frac)
    tmp["_year"] = years.fillna(years.median())
    group_year = tmp.groupby(group_col)["_year"].median().sort_values()
    n = len(group_year)
    train_cut = int(n * (1.0 - val_frac - test_frac))
    val_cut = int(n * (1.0 - test_frac))
    assignment = {g: "train" for g in group_year.index[:train_cut]}
    assignment.update({g: "val" for g in group_year.index[train_cut:val_cut]})
    assignment.update({g: "test" for g in group_year.index[val_cut:]})
    return tmp[group_col].map(assignment)


def add_protocol_splits(
    df: pd.DataFrame,
    seed: int = 2027,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, list[SplitReport]]:
    out = df.copy()
    if "leakage_group" not in out:
        out["leakage_group"] = leakage_groups(out)

    out["split_s0"] = _assign_random_groups(out, "leakage_group", seed, val_frac, test_frac)
    out["_group_s1"] = connected_groups(out, ["parent_inchikey", "document_chembl_id", "scaffold"], name="_group_s1")
    out["_group_s2"] = connected_groups(out, ["parent_inchikey", "document_chembl_id", "assay_chembl_id"], name="_group_s2")
    out["split_s1"] = _assign_random_groups(out, "_group_s1", seed, val_frac, test_frac)
    out["split_s2"] = _assign_random_groups(out, "_group_s2", seed + 1, val_frac, test_frac)
    out["split_s3"] = _assign_temporal(out, "leakage_group", val_frac, test_frac)

    out["_group_s4"] = connected_groups(
        out,
        ["parent_inchikey", "document_chembl_id", "scaffold", "assay_chembl_id"],
        name="_group_s4",
    )
    out["split_s4"] = _assign_random_groups(out, "_group_s4", seed + 2, val_frac, test_frac)
    out = out.drop(columns=["_group_s1", "_group_s2", "_group_s4"])

    reports = [split_leakage_report(out, col) for col in ["split_s0", "split_s1", "split_s2", "split_s3", "split_s4"]]
    return out, reports


def split_leakage_report(df: pd.DataFrame, split_col: str) -> SplitReport:
    train = df[df[split_col] == "train"]
    val = df[df[split_col] == "val"]
    test = df[df[split_col] == "test"]
    train_mol = set(train["parent_inchikey"].dropna().astype(str))
    test_mol = set(test["parent_inchikey"].dropna().astype(str))
    train_doc = set(train["document_chembl_id"].dropna().astype(str))
    test_doc = set(test["document_chembl_id"].dropna().astype(str))
    return SplitReport(
        split=split_col.upper().replace("SPLIT_", ""),
        train_rows=len(train),
        val_rows=len(val),
        test_rows=len(test),
        inchikey_train_test_overlap=len(train_mol & test_mol),
        document_train_test_overlap=len(train_doc & test_doc),
    )
