"""
Summarize fine-tuning runs: best validation macro-F1 (and its epoch) and test scores.

Usage: python scripts/collect_results.py [logs_dir] [--csv out.csv]
"""

import argparse
import csv
import glob
import os
import re


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _best_validation(rows: list[dict]) -> dict:
    """Epoch with the best validation macro-F1 and its scores."""
    vals = [r for r in rows if r.get("val_macro-f1")]
    if not vals:
        return {}
    best = max(vals, key=lambda r: float(r["val_macro-f1"]))
    return {
        "best_epoch": int(best["epoch"]),
        "epochs": max(int(r["epoch"]) for r in rows if r.get("epoch")) + 1,
        "val_macro_f1": float(best["val_macro-f1"]),
        "val_micro_f1": float(best["val_micro-f1"]),
    }


def _test(rows: list[dict]) -> dict:
    """Scores of the last test row."""
    tests = [r for r in rows if r.get("test_macro-f1")]
    if not tests:
        return {}
    return {
        "test_macro_f1": float(tests[-1]["test_macro-f1"]),
        "test_micro_f1": float(tests[-1]["test_micro-f1"]),
    }


def collect(logs_dir: str) -> list[dict]:
    """One result row per run in a log directory."""
    results = []
    for run_dir in sorted(glob.glob(os.path.join(logs_dir, "*"))):
        # A continued run (e.g. 100 -> 200 epochs) writes further version directories,
        # they count as one run with the test of the newest version
        versions = sorted(
            (
                v
                for v in glob.glob(os.path.join(run_dir, "version_*"))
                if re.fullmatch(r"version_\d+", os.path.basename(v))
            ),
            key=lambda v: int(v.rsplit("_", 1)[-1]),
        )
        rows = []
        for version in versions:
            metrics = os.path.join(version, "metrics.csv")
            if os.path.isfile(metrics):
                rows += _read(metrics)
        if not rows:
            continue
        row = {
            "run": os.path.basename(run_dir),
            "version": os.path.basename(versions[-1]),
        }
        row.update(_best_validation(rows))
        for version in reversed(versions):
            test_metrics = os.path.join(version + "_test", "metrics.csv")
            if os.path.isfile(test_metrics):
                row.update(_test(_read(test_metrics)))
                break
        if "val_macro_f1" in row:
            results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", nargs="?", default="logs")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    results = collect(args.logs_dir)
    columns = [
        "run",
        "version",
        "epochs",
        "best_epoch",
        "val_macro_f1",
        "val_micro_f1",
        "test_macro_f1",
        "test_micro_f1",
    ]
    print("\t".join(columns))
    for r in results:
        print(
            "\t".join(
                f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in columns
            )
        )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
