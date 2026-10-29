"""
Chemical composition of the pretraining data compared with ChEBI.

Takes the first N training molecules of the PubChem stream (the fixed random order in which
the pretraining read them) and all molecules of a ChEBI dataset, and writes for every
molecule the descriptors of analyse_predictions.py (size, rings, charge, presence of the
RDKit fr_* groups):

  pubchem.csv  position in the stream, PubChem CID, descriptors
  chebi.csv    chebi_id, split, descriptors

Usage:
  python scripts/pubchem_composition.py --pubchem data/Pubchem/2026-09-01 \\
      --processed data/chebi_v241/ChEBI50/processed --n 200000 --out logs/analysis/composition
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from chebi_utils.read_molecule import smiles_or_inchi_to_mol
from rdkit import RDLogger

from analyse_predictions import descriptors


def pubchem_row(item: tuple[int, str, str]) -> dict:
    """Descriptors of a PubChem molecule, given as (stream position, CID, SMILES)."""
    position, cid, smiles = item
    RDLogger.DisableLog("rdApp.*")
    mol = smiles_or_inchi_to_mol(smiles)
    row = {"position": position, "cid": cid}
    if mol is not None:
        row.update(descriptors(mol))
    return row


def chebi_row(item: tuple) -> dict:
    """Descriptors of a ChEBI molecule, given as (ChEBI id, split, RDKit molecule)."""
    chebi_id, split, mol = item
    row = {"chebi_id": chebi_id, "split": split}
    if mol is not None:
        row.update(descriptors(mol))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pubchem",
        required=True,
        help="PubChem release dir with raw/full/smiles.txt and stream/",
    )
    parser.add_argument(
        "--processed",
        required=True,
        help="processed ChEBI dir with data.pkl and splits.csv",
    )
    parser.add_argument(
        "--n", type=int, default=200000, help="number of PubChem molecules"
    )
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=10000,
        help="held-out start of the stream",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stream = os.path.join(args.pubchem, "stream")
    order = np.load(
        os.path.join(stream, f"permutation_seed{args.stream_seed}.npy"), mmap_mode="r"
    )
    offsets = np.load(os.path.join(stream, "line_offsets.npy"), mmap_mode="r")
    items = []
    with open(os.path.join(args.pubchem, "raw", "full", "smiles.txt"), "rb") as f:
        for position in range(args.validation_size, args.validation_size + args.n):
            f.seek(int(offsets[order[position]]))
            cid, _, smiles = f.readline().decode().rstrip("\n").partition("\t")
            items.append((position, cid, smiles))
    with Pool(args.workers) as pool:
        pubchem = pd.DataFrame(pool.map(pubchem_row, items, chunksize=500))
    pubchem.to_csv(os.path.join(args.out, "pubchem.csv"), index=False)

    data = pd.read_pickle(os.path.join(args.processed, "data.pkl"))
    data["chebi_id"] = data["chebi_id"].astype(int)
    split = pd.read_csv(os.path.join(args.processed, "splits.csv")).set_index("id")[
        "split"
    ]
    items = [
        (cid, split.get(cid, ""), mol)
        for cid, mol in zip(data["chebi_id"], data["mol"])
    ]
    with Pool(args.workers) as pool:
        chebi = pd.DataFrame(pool.map(chebi_row, items, chunksize=500))
    chebi.to_csv(os.path.join(args.out, "chebi.csv"), index=False)
    print(
        f"{len(pubchem)} PubChem molecules ({pubchem['smiles'].notna().sum()} parsed), "
        f"{len(chebi)} ChEBI molecules -> {args.out}"
    )


if __name__ == "__main__":
    main()
