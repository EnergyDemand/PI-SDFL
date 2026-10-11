"""
aggregate_wind_results.py — Aggregate all wind AEI clean runs into one summary.

Reads per-seed final_metrics.json from results/wind/clean/<Method>/seed_*/
and writes results/wind/clean/summary_wind_clean.json.

Run once after all runners have completed:
    python aggregate_wind_results.py

Methods expected:
    Local-only
    Centralized
    Vanilla-Sequential
    CWC
    PI-Vanilla-Sequential
    FedAvg
    PI-SDFL-full
    Frozen-Reference
"""
import json, os, sys
import numpy as np
from typing import Dict, List

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
CLEAN_DIR   = os.path.join(PROJECT_DIR, "results", "wind", "clean")

METHODS = [
    "Local-only",
    "Centralized",
    "Vanilla-Sequential",
    "CWC",
    "PI-Vanilla-Sequential",
    "FedAvg",
    "PI-SDFL-full",
    "Frozen-Reference",
]


def safe_folder(method: str) -> str:
    return (method.replace(" ", "_").replace("/", "-")
                  .replace("(", "").replace(")", ""))


def load_seed_metrics(method: str) -> List[Dict]:
    folder = os.path.join(CLEAN_DIR, safe_folder(method))
    if not os.path.isdir(folder):
        print(f"  WARNING: folder not found — {folder}")
        return []
    results = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry, "final_metrics.json")
        if os.path.isfile(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def aggregate(results: List[Dict]) -> Dict:
    if not results:
        return {}
    summary = {}
    for k in results[0]:
        vals = [float(r[k]) for r in results
                if k in r and r[k] is not None
                and isinstance(r[k], (int, float))
                and np.isfinite(float(r[k]))]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def main() -> None:
    print(f"\nAggregating wind AEI results from: {CLEAN_DIR}\n")
    summary = {}
    for method in METHODS:
        results = load_seed_metrics(method)
        if results:
            summary[method] = aggregate(results)
            n    = len(results)
            r2   = summary[method].get("global_R2", {})
            pvr  = summary[method].get("global_mono_viol_rate", {})
            r2_s = f"{r2['mean']:.4f}±{r2['std']:.4f}" if r2 else "N/A (Local-only)"
            pvr_s= f"{pvr['mean']:.4f}" if pvr else "--"
            print(f"  {method:<28} n={n}  global_R2={r2_s}  PVR={pvr_s}")
        else:
            print(f"  {method:<28} NO DATA FOUND")

    out = os.path.join(CLEAN_DIR, "summary_wind_clean.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
