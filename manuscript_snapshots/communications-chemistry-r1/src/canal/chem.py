from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")


EXACT_RELATIONS = {"=", "~"}
LOWER_BOUND_RELATIONS = {">", ">="}
UPPER_BOUND_RELATIONS = {"<", "<="}


@dataclass(frozen=True)
class ActivityValue:
    p_activity: float
    p_relation: str


def normalize_relation(relation: object) -> str | None:
    if relation is None:
        return None
    rel = str(relation).strip()
    if rel in EXACT_RELATIONS:
        return "="
    if rel in LOWER_BOUND_RELATIONS:
        return ">"
    if rel in UPPER_BOUND_RELATIONS:
        return "<"
    return None


def nm_to_p_activity(value_nm: object, relation: object) -> ActivityValue | None:
    """Convert nM relation/value to pActivity, reversing inequality direction.

    ChEMBL relations are attached to standard_value in nM. CANAL trains on
    pActivity = 9 - log10(nM), so a nM lower bound (`> 1000 nM`) becomes a
    pActivity upper bound (`< 6`).
    """

    rel = normalize_relation(relation)
    if rel is None:
        return None
    try:
        value = float(value_nm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    p_value = 9.0 - math.log10(value)
    if rel == ">":
        p_rel = "<"
    elif rel == "<":
        p_rel = ">"
    else:
        p_rel = "="
    return ActivityValue(p_activity=p_value, p_relation=p_rel)


def parent_inchikey(inchikey: object) -> str | None:
    if inchikey is None:
        return None
    text = str(inchikey).strip()
    if not text:
        return None
    return text.split("-")[0]


def canonicalize_smiles(smiles: object) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_smiles(smiles: object) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None
    return scaffold or "NO_SCAFFOLD"


def ecfp4(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr
