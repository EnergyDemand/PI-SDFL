"""
aggregate_clean_results.py — Combine all clean-condition final runs into one summary.

Reads per-seed final_metrics.json from results_new/final/<Method>/seed_*/
and writes results_new/final/summary_clean.json with mean ± std per method.

Run once after all clean runners and ablation runners have completed:
    python aggregate_clean_results.py

Methods expected (8 clean comparison + 5 ablation = 13 total):
    Local-only
    Centralized
    Vanilla-Sequential
    CWC
    PI-Vanilla-Sequential
    PI-SDFL-full
    FedAvg
    Frozen-Reference
    PI-SDFL-no-anchor
    PI-SDFL-no-gate
    PI-SDFL-no-mono
    PI-SDFL-no-warmstart
    PI-SDFL-no-gate-no-mono
"""

import os
import sys
import json
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
FINAL_DIR   = os.path.join(PROJECT_DIR, "results", "pem", "final")

METHODS = [
    "Local-only",
    "Centralized",
    "Vanilla-Sequential",
    "CWC",
    "PI-Vanilla-Sequential",
    "PI-SDFL-full",
    "FedAvg",
    "Frozen-Reference",
    "PI-SDFL-no-anchor",
    "PI-SDFL-no-gate",
    "PI-SDFL-no-mono",
    "PI-SDFL-no-warmstart",
    "PI-SDFL-no-gate-no-mono",
]


def safe_folder_name(method: str) -> str:
    return (method.replace(" ", "_").replace("/", "-")
                  .replace("(", "").replace(")", ""))


def load_seed_metrics(method: str) -> List[Dict]:
    folder = os.path.join(FINAL_DIR, safe_folder_name(method))
    if not os.path.isdir(folder):
        print(f"  WARNING: folder not found — {folder}")
        return []
    results = []
    for entry in sorted(os.listdir(folder)):
        seed_dir = os.path.join(folder, entry)
        metrics_path = os.path.join(seed_dir, "final_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                results.append(json.load(f))
    return results


def aggregate(results: List[Dict]) -> Dict:
    if not results:
        return {}
    summary = {}
    for k in results[0]:
        vals = [
            float(r[k]) for r in results
            if k in r and r[k] is not None
            and isinstance(r[k], (int, float))
            and np.isfinite(float(r[k]))
        ]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def main() -> None:
    print(f"\nAggregating clean final results from: {FINAL_DIR}\n")
    summary = {}
    for method in METHODS:
        results = load_seed_metrics(method)
        if results:
            summary[method] = aggregate(results)
            n = len(results)
            r2 = summary[method].get("global_R2", {})
            r2_str = (f"{r2['mean']:.4f}±{r2['std']:.4f}"
                      if r2 else "N/A (Local-only)")
            print(f"  {method:<30} n={n}  global_R2={r2_str}")
        else:
            print(f"  {method:<30} NO DATA FOUND")

    out_path = os.path.join(FINAL_DIR, "summary_clean.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
