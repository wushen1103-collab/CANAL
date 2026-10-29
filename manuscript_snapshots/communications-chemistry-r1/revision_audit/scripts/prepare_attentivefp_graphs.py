#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ATOM_SYMBOLS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYDROGENS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
STEREOS = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]


def one_hot(value: object, choices: list[object]) -> list[int]:
    return [int(value == choice) for choice in choices] + [int(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[int]:
    return (
        one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_hot(atom.GetTotalDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot(atom.GetTotalNumHs(), HYDROGENS)
        + one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[int]:
    return (
        one_hot(str(bond.GetBondType()), BOND_TYPES)
        + one_hot(str(bond.GetStereo()), STEREOS)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    )


def featurize(item: tuple[int, str]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    index, smiles = item
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES at unique index {index}: {smiles}")
    nodes = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.uint8)
    edge_pairs: list[tuple[int, int]] = []
    edge_values: list[list[int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.extend([(begin, end), (end, begin)])
        edge_values.extend([features, features])
    if edge_pairs:
        edges = np.asarray(edge_pairs, dtype=np.int32).T
        edge_features = np.asarray(edge_values, dtype=np.uint8)
    else:
        edges = np.empty((2, 0), dtype=np.int32)
        edge_features = np.empty((0, len(BOND_TYPES) + 1 + len(STEREOS) + 1 + 2), dtype=np.uint8)
    return index, nodes, edges, edge_features


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset, columns=["canonical_smiles_rdkit"])
    row_graph_index, unique_smiles = pd.factorize(df["canonical_smiles_rdkit"].astype(str), sort=True)
    tasks = list(enumerate(unique_smiles.tolist()))
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, (index, nodes, edges, edge_features) in enumerate(
            executor.map(featurize, tasks, chunksize=64), start=1
        ):
            results[index] = (nodes, edges, edge_features)
            if completed % 10_000 == 0:
                print(f"featurized {completed}/{len(tasks)}", flush=True)

    if any(result is None for result in results):
        raise RuntimeError("Incomplete graph featurization")
    graph_data = [result for result in results if result is not None]
    node_ptr = np.zeros(len(graph_data) + 1, dtype=np.int64)
    edge_ptr = np.zeros(len(graph_data) + 1, dtype=np.int64)
    node_ptr[1:] = np.cumsum([len(nodes) for nodes, _, _ in graph_data])
    edge_ptr[1:] = np.cumsum([edges.shape[1] for _, edges, _ in graph_data])
    nodes = np.concatenate([item[0] for item in graph_data], axis=0)
    edges = np.concatenate([item[1] for item in graph_data], axis=1)
    edge_features = np.concatenate([item[2] for item in graph_data], axis=0)

    arrays = {
        "node_features.npy": nodes,
        "edge_index.npy": edges,
        "edge_features.npy": edge_features,
        "node_ptr.npy": node_ptr,
        "edge_ptr.npy": edge_ptr,
        "row_graph_index.npy": row_graph_index.astype(np.int32),
    }
    for name, array in arrays.items():
        np.save(args.output_dir / name, array, allow_pickle=False)
    pd.DataFrame({"graph_index": np.arange(len(unique_smiles)), "smiles": unique_smiles}).to_csv(
        args.output_dir / "unique_smiles.csv", index=False
    )

    paths = [args.output_dir / name for name in arrays] + [args.output_dir / "unique_smiles.csv"]
    manifest = {
        "dataset": str(args.dataset),
        "rows": int(len(df)),
        "unique_graphs": int(len(unique_smiles)),
        "nodes": int(len(nodes)),
        "directed_edges": int(edges.shape[1]),
        "node_feature_dim": int(nodes.shape[1]),
        "edge_feature_dim": int(edge_features.shape[1]),
        "workers": int(args.workers),
        "feature_definition": {
            "atom": "symbol, total degree, formal charge, total H, hybridization, aromatic, ring",
            "bond": "bond type, stereo, conjugated, ring; both directions",
        },
        "sha256": {path.name: file_sha256(path) for path in paths},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
