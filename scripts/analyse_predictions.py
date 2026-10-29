"""
Analyze the saved test predictions of the fine-tuning runs per class and per molecule.

For every run with a test_predictions.pt (written by TestPredictionWriter), the script
recomputes micro- and macro-F1 as chebai does (threshold 0.5, macro-F1 averaged
over the classes with at least one positive test molecule), also on the subset of test
molecules that are fully curated in ChEBI ("3 star"), and writes long-format tables:

  summary.csv       run, dataset, graph, strategy, seed, micro/macro-F1 on all and on 3-star molecules
  per_class.csv     run, class_id, tp, fp, fn
  per_molecule.csv  run, chebi_id, fp, fn
  classes.csv       class_id, name, train_count, test_count, test_count_3star, is_3star
  molecules.csv     chebi_id, smiles, size and ring descriptors, 3-star flag, number of
                    labels, presence of the 85 RDKit functional group counters (fr_*)
  test_labels.csv   chebi_id, ";"-joined class ids of the molecule

Usage:
  python scripts/analyse_predictions.py --logs logs --processed data/chebi_v241/ChEBI50/processed \\
      --obo data/chebi_v241/raw/chebi.obo --runs 'hpc_*' --out logs/analysis/chebi50
"""

import argparse
import fnmatch
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors


def parse_obo(path: str) -> pd.DataFrame:
    """Id (int), name and 3-star flag of every ChEBI term."""
    rows, term = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if term:
                    rows.append(term)
                term = {"id": None, "name": "", "three_star": False}
            elif term is None:
                continue
            elif line.startswith("id: CHEBI:"):
                term["id"] = int(line[len("id: CHEBI:") :])
            elif line.startswith("name: "):
                term["name"] = line[len("name: ") :]
            elif line == "subset: 3_STAR":
                term["three_star"] = True
            elif line.startswith("[") and line != "[Term]":
                rows.append(term)
                term = None
    if term:
        rows.append(term)
    return pd.DataFrame([r for r in rows if r and r["id"] is not None]).set_index("id")


def f1_scores(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[float, float]:
    """
    Micro- and macro-F1 as in chebai; classes without positive labels are left out of
    the macro average.
    """
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    support = tp + fn
    mask = support > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp[mask] / (tp[mask] + fp[mask])
        recall = tp[mask] / support[mask]
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return float(micro), float(f1.mean()) if f1.size else float("nan")


def counts(
    pred: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, false positives and false negatives per class."""
    tp = (pred & labels).sum(axis=0)
    fp = (pred & ~labels).sum(axis=0)
    fn = (~pred & labels).sum(axis=0)
    return tp, fp, fn


def describe(run: str) -> dict:
    """Dataset, graph, strategy, variant and seed of a run, e.g. hpc_augmented_all_s1."""
    m = re.match(r"(hpc\d*)_(flat|augmented)_([a-z]+)(?:_(\w+?))?_s(\d+)$", run)
    if not m:
        return {"dataset": "", "graph": "", "strategy": "", "variant": "", "seed": -1}
    prefix, graph, strategy, variant, seed = m.groups()
    return {
        "dataset": "chebi25" if prefix == "hpc25" else "chebi50",
        "graph": graph,
        "strategy": strategy,
        "variant": variant or "",
        "seed": int(seed),
    }


def prediction_file(run_dir: str) -> str | None:
    """test_predictions.pt of the last test run, otherwise of the last predict run."""
    files = sorted(
        glob.glob(os.path.join(run_dir, "version_*_predict", "test_predictions.pt"))
    )
    files += sorted(
        glob.glob(os.path.join(run_dir, "version_*_test", "test_predictions.pt"))
    )
    return files[-1] if files else None


def descriptors(mol) -> dict:
    """SMILES, size, ring and charge descriptors and fr_* groups of a molecule."""
    row = {
        "smiles": Chem.MolToSmiles(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "mol_weight": Descriptors.MolWt(mol),
    }
    for name, fn in sorted(
        (n, f)
        for n, f in vars(Fragments).items()
        if n.startswith("fr_") and callable(f)
    ):
        try:
            row[name] = int(fn(mol) > 0)
        except Exception:  # a few RDKit counters fail on unusual molecules
            row[name] = -1
    return row


def molecule_table(
    data: pd.DataFrame, ids: list[int], three_star: set[int]
) -> pd.DataFrame:
    """Descriptor table of the test molecules."""
    rows = []
    mols = data.set_index("chebi_id")["mol"]
    for cid in ids:
        mol = mols.get(cid)
        row = {"chebi_id": cid, "three_star": cid in three_star}
        if mol is not None:
            row.update(descriptors(mol))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument(
        "--processed",
        required=True,
        help="processed dir with classes.txt, splits.csv, data.pkl",
    )
    parser.add_argument("--obo", required=True)
    parser.add_argument("--runs", default="hpc_*", help="glob pattern of the run names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    classes = [
        int(c) for c in open(os.path.join(args.processed, "classes.txt")).read().split()
    ]
    obo = parse_obo(args.obo)
    three_star = set(obo.index[obo["three_star"]])
    splits = pd.read_csv(os.path.join(args.processed, "splits.csv")).set_index("id")[
        "split"
    ]
    data = pd.read_pickle(os.path.join(args.processed, "data.pkl"))
    # data.pkl stores the ids as strings, splits.csv and the predictions as integers
    data["chebi_id"] = data["chebi_id"].astype(int)
    label_columns = [str(c) for c in classes]

    summary, per_class, per_molecule = [], [], []
    test_ids, test_labels = None, None
    for run_dir in sorted(glob.glob(os.path.join(args.logs, "*"))):
        run = os.path.basename(run_dir)
        if not fnmatch.fnmatch(run, args.runs) or "pretrain" in run:
            continue
        path = prediction_file(run_dir)
        if path is None:
            print(f"no predictions for {run}")
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        probs = saved["probs"].float().numpy()
        labels = saved["labels"].numpy().astype(bool)
        ids = [int(i) for i in saved["idents"]]
        order = np.argsort(ids)
        probs, labels, ids = probs[order], labels[order], [ids[i] for i in order]
        if test_ids is None:
            test_ids, test_labels = ids, labels
        elif ids != test_ids:
            raise ValueError(f"{run}: different test molecules than the first run")
        pred = probs > 0.5
        tp, fp, fn = counts(pred, labels)
        micro, macro = f1_scores(tp, fp, fn)
        star = np.array([i in three_star for i in ids])
        s_tp, s_fp, s_fn = counts(pred[star], labels[star])
        s_micro, s_macro = f1_scores(s_tp, s_fp, s_fn)
        summary.append(
            {
                "run": run,
                **describe(run),
                "n_test": len(ids),
                "micro_f1": micro,
                "macro_f1": macro,
                "n_test_3star": int(star.sum()),
                "micro_f1_3star": s_micro,
                "macro_f1_3star": s_macro,
                "file": os.path.relpath(path, args.logs),
            }
        )
        per_class.append(
            pd.DataFrame(
                {"run": run, "class_id": classes, "tp": tp, "fp": fp, "fn": fn}
            )
        )
        per_molecule.append(
            pd.DataFrame(
                {
                    "run": run,
                    "chebi_id": ids,
                    "fp": (pred & ~labels).sum(axis=1),
                    "fn": (~pred & labels).sum(axis=1),
                }
            )
        )
        print(
            f"{run}: micro {micro:.4f} macro {macro:.4f} | 3-star ({star.sum()}) micro {s_micro:.4f} macro {s_macro:.4f}"
        )

    if not summary:
        raise SystemExit("no predictions found")
    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.concat(per_class).to_csv(os.path.join(args.out, "per_class.csv"), index=False)
    pd.concat(per_molecule).to_csv(
        os.path.join(args.out, "per_molecule.csv"), index=False
    )

    train_ids = set(splits.index[splits == "train"])
    in_train = data["chebi_id"].isin(train_ids).to_numpy()
    train_count = data.loc[in_train, label_columns].sum().to_numpy()
    star = np.array([i in three_star for i in test_ids])
    pd.DataFrame(
        {
            "class_id": classes,
            "name": [obo["name"].get(c, "") for c in classes],
            "is_3star": [c in three_star for c in classes],
            "train_count": train_count,
            "test_count": test_labels.sum(axis=0),
            "test_count_3star": test_labels[star].sum(axis=0),
        }
    ).to_csv(os.path.join(args.out, "classes.csv"), index=False)
    molecule_table(data, test_ids, three_star).to_csv(
        os.path.join(args.out, "molecules.csv"), index=False
    )
    pd.DataFrame(
        {
            "chebi_id": test_ids,
            "class_ids": [
                ";".join(str(classes[j]) for j in np.flatnonzero(row))
                for row in test_labels
            ],
        }
    ).to_csv(os.path.join(args.out, "test_labels.csv"), index=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
