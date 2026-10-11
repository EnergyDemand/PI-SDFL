"""
run_nlr_preprocessing.py
========================
NLR PEM preprocessing pipeline for PI-SDFL.

Place this file in:
    D:\\GPU\\pi_sdfl_v2\\src_nlr\\run_nlr_preprocessing.py

Run:
    cd /d D:\\GPU\\pi_sdfl_v2
    python src_nlr\\run_nlr_preprocessing.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLIENT_SOURCES = {
    1: "combined_historical_wind_experiments.csv",
    2: "combined_solarPV_experiments.csv",
    3: "combined_wind_experiments.csv",
    4: "combined_simulated_tidal_experiments.csv",
}

CLIENT_LABELS = {
    1: "historical_wind",
    2: "historical_solar_pv",
    3: "simulated_wind",
    4: "simulated_tidal",
}

# NLR raw channel -> model-facing feature name
FEATURE_MAP = {
    "H2E_f_PSU_A_Current":  "current",
    "H2E_f_TE218_Temp":     "temp_in",
    "H2E_f_TE219_Temp":     "temp_return",
    "H2E_f_PT264_Pressure": "pressure_in",
    "H2E_f_PT307_Pressure": "pressure_separator",
    "H2E_f_PT312_Pressure": "pressure_product",
}

FEATURES        = list(FEATURE_MAP.values())
CURRENT_FEATURE = "current"
CONTEXT_FEATURES = [f for f in FEATURES if f != CURRENT_FEATURE]
TARGET          = "h2_flow_rate"

# Some files include the unit in the column name; others do not.
TARGET_CANDIDATES = [
    "IVAL_f_FM011_Flow (kg/hr)",
    "IVAL_f_FM011_Flow",
]

# The experiment identifier column name varies across source files.
# Priority: use whichever of these is present.
EXPERIMENT_COL_CANDIDATES = ["Experiment", "Wind Turbine"]

# Temporal preprocessing
LAG_SECONDS         = 3
AVERAGE_SECONDS     = 10
MIN_VALID_PER_WINDOW = 8   # minimum valid 1-Hz obs to keep a 10-s window

# Chronological split fractions
FRAC_TRAIN, FRAC_CALIB, FRAC_AUDIT, FRAC_TEST = 0.65, 0.10, 0.10, 0.15
EMBARGO_SECONDS = 60

# Physics probe
PROBE_CURRENT_POINTS   = 50
PROBE_CONTEXT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
PROBE_CURRENT_Q_LOW    = 0.01
PROBE_CURRENT_Q_HIGH   = 0.99


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dirs(project_dir: Path) -> Dict[str, Path]:
    paths = {
        "raw":        project_dir / "data" / "nlr_raw",
        "processed":  project_dir / "data" / "nlr_processed",
        "clients":    project_dir / "data" / "nlr_processed" / "clients_aligned_10s",
        "splits_raw": project_dir / "data" / "nlr_processed" / "splits_raw",
        "splits_norm":project_dir / "data" / "nlr_processed" / "splits_normalized",
        "metadata":   project_dir / "data" / "nlr_processed" / "metadata",
        "results":    project_dir / "results_nlr",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_experiment_col(df: pd.DataFrame, source_name: str) -> str:
    for col in EXPERIMENT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"{source_name}: no experiment column found. "
        f"Looked for: {EXPERIMENT_COL_CANDIDATES}"
    )


def detect_target_col(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"H2 flow column not found. Expected one of: {TARGET_CANDIDATES}"
    )


def validate_columns(df: pd.DataFrame, source_name: str) -> Tuple[str, str]:
    missing = [c for c in FEATURE_MAP if c not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: missing feature columns: {missing}")
    exp_col    = detect_experiment_col(df, source_name)
    target_col = detect_target_col(df)
    return exp_col, target_col


# ---------------------------------------------------------------------------
# 3-second alignment + 10-second averaging (per experiment)
# ---------------------------------------------------------------------------

def prepare_one_experiment(
    g: pd.DataFrame,
    target_col: str,
    experiment_name: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Align process inputs at t with H2 flow at t+3 s, then
    aggregate into non-overlapping 10-s windows.

    Missing observations do NOT compress time; windows are based on
    original 1-Hz row positions.
    """
    g = g.reset_index(drop=True).copy()
    n_raw = len(g)

    x = pd.DataFrame(index=g.index)
    for raw_name, short_name in FEATURE_MAP.items():
        x[short_name] = pd.to_numeric(g[raw_name], errors="coerce")
    y = pd.to_numeric(g[target_col], errors="coerce")

    x[TARGET]         = y.shift(-LAG_SECONDS)
    x["_row_position"] = np.arange(len(x), dtype=np.int64)
    x["_window"]       = x["_row_position"] // AVERAGE_SECONDS

    core = FEATURES + [TARGET]
    x["_valid"] = np.isfinite(x[core]).all(axis=1)

    rows = []
    for window_id, w in x.groupby("_window", sort=True):
        valid = w.loc[w["_valid"], core]
        if len(valid) < MIN_VALID_PER_WINDOW:
            continue
        row = valid.mean(axis=0).to_dict()
        row["experiment"]    = experiment_name
        row["window_id"]     = int(window_id)
        row["valid_seconds"] = int(len(valid))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[["experiment", "window_id", "valid_seconds"] + FEATURES + [TARGET]]

    info = {
        "experiment":               experiment_name,
        "raw_rows":                 int(n_raw),
        "aligned_rows_possible":    int(max(n_raw - LAG_SECONDS, 0)),
        "ten_second_windows_kept":  int(len(out)),
        "ten_second_windows_dropped": int(
            math.ceil(n_raw / AVERAGE_SECONDS) - len(out)
        ),
    }
    return out, info


# ---------------------------------------------------------------------------
# Client-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_client(
    csv_path: Path, client_id: int
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    print(f"\n  Reading {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    exp_col, target_col = validate_columns(df, csv_path.name)
    raw_rows = len(df)

    # Client 3 (combined_wind) contains characterization_200 — must be removed.
    n_char_removed = 0
    if client_id == 3:
        mask = df[exp_col].astype(str).str.contains(
            "characterization", case=False, na=False
        )
        n_char_removed = int(mask.sum())
        if n_char_removed:
            print(f"    Removing {n_char_removed} characterization rows.")
            df = df.loc[~mask].copy()

    experiments, experiment_info = [], []
    for exp_name, g in df.groupby(exp_col, sort=False, dropna=False):
        processed, info = prepare_one_experiment(
            g=g, target_col=target_col, experiment_name=str(exp_name)
        )
        if len(processed):
            experiments.append(processed)
        experiment_info.append(info)

    if not experiments:
        raise RuntimeError(f"{csv_path.name}: no usable experiments.")

    out = pd.concat(experiments, ignore_index=True)
    client_info = {
        "client_id":                     client_id,
        "client_label":                  CLIENT_LABELS[client_id],
        "source_file":                   csv_path.name,
        "source_sha256_12":              sha256_short(csv_path),
        "experiment_col_used":           exp_col,
        "raw_rows":                      int(raw_rows),
        "characterization_rows_removed": n_char_removed,
        "processed_10s_rows":            int(len(out)),
        "n_experiments":                 int(out["experiment"].nunique()),
    }
    return out, experiment_info, client_info


# ---------------------------------------------------------------------------
# Chronological 65/10/10/15 split with 60-s embargo
# ---------------------------------------------------------------------------

def split_one_experiment(
    g: pd.DataFrame, embargo_bins: int
) -> Dict[str, pd.DataFrame]:
    g = g.sort_values("window_id").reset_index(drop=True)
    n = len(g)
    total_gap = 3 * embargo_bins

    if n <= total_gap + 20:
        raise ValueError(
            f"Experiment '{g['experiment'].iloc[0]}' has only {n} 10-s windows — "
            "too short for the requested split + embargo protocol."
        )

    usable  = n - total_gap
    n_train = int(np.floor(FRAC_TRAIN * usable))
    n_calib = int(np.floor(FRAC_CALIB * usable))
    n_audit = int(np.floor(FRAC_AUDIT * usable))
    n_test  = usable - n_train - n_calib - n_audit

    i = 0
    train = g.iloc[i : i + n_train].copy(); i += n_train + embargo_bins
    calib = g.iloc[i : i + n_calib].copy(); i += n_calib + embargo_bins
    audit = g.iloc[i : i + n_audit].copy(); i += n_audit + embargo_bins
    test  = g.iloc[i : i + n_test ].copy()

    return {"train": train, "calib": calib, "audit": audit, "test": test}


def chronological_split_client(
    df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    embargo_bins = int(np.ceil(EMBARGO_SECONDS / AVERAGE_SECONDS))
    buckets: Dict[str, List] = {"train": [], "calib": [], "audit": [], "test": []}
    details = []

    for exp_name, g in df.groupby("experiment", sort=False):
        s = split_one_experiment(g, embargo_bins)
        for part in buckets:
            buckets[part].append(s[part])
        details.append({
            "experiment":                       str(exp_name),
            "processed_rows":                   int(len(g)),
            "train":                            int(len(s["train"])),
            "calib":                            int(len(s["calib"])),
            "audit":                            int(len(s["audit"])),
            "test":                             int(len(s["test"])),
            "embargo_bins_each_boundary":       embargo_bins,
            "embargo_seconds_each_boundary":    embargo_bins * AVERAGE_SECONDS,
        })

    splits = {
        part: pd.concat(frames, ignore_index=True)
        for part, frames in buckets.items()
    }
    return splits, details


# ---------------------------------------------------------------------------
# Standardizer (fit on Client 1 training data only)
# ---------------------------------------------------------------------------

def fit_standardizer(client1_train: pd.DataFrame) -> Dict:
    X = client1_train[FEATURES].to_numpy(dtype=np.float64)
    y = client1_train[TARGET].to_numpy(dtype=np.float64)

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0, ddof=0)   # population std — consistent with PI-SDFL
    y_mean = float(y.mean())
    y_std  = float(y.std(ddof=0))

    bad_X = [FEATURES[i] for i, s in enumerate(X_std) if s < 1e-12]
    if bad_X:
        raise ValueError(f"Near-zero training std for features: {bad_X}")
    if y_std < 1e-12:
        raise ValueError("Near-zero target std.")

    current_idx = FEATURES.index(CURRENT_FEATURE)
    return {
        "features": FEATURES,
        "target":   TARGET,
        "X_mean":   {f: float(v) for f, v in zip(FEATURES, X_mean)},
        "X_std":    {f: float(v) for f, v in zip(FEATURES, X_std)},
        "y_mean":   y_mean,
        "y_std":    y_std,
        "I_mean":   float(X_mean[current_idx]),
        "I_std":    float(X_std[current_idx]),
        "fit_source": "Client 1 (historical wind) training split only",
        "ddof": 0,
    }


def normalize_split(df: pd.DataFrame, standardizer: Dict) -> pd.DataFrame:
    out = df[["experiment", "window_id", "valid_seconds"]].copy()
    for feature in FEATURES:
        out[feature] = (
            df[feature].astype(float) - standardizer["X_mean"][feature]
        ) / standardizer["X_std"][feature]
    out[TARGET] = (
        df[TARGET].astype(float) - standardizer["y_mean"]
    ) / standardizer["y_std"]
    return out


# ---------------------------------------------------------------------------
# Multi-context physics probe
# ---------------------------------------------------------------------------

def build_multicontext_probe(
    pooled_train: pd.DataFrame, standardizer: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Five empirical operating contexts from pooled training data nearest to
    the q10/q25/q50/q75/q90 non-current quantile vectors; current is swept
    independently within each context (50 points per context = 250 total).
    """
    ctx_data  = pooled_train[CONTEXT_FEATURES].astype(float)
    ctx_array = ctx_data.to_numpy(dtype=float)
    ctx_scale = np.array(
        [standardizer["X_std"][f] for f in CONTEXT_FEATURES], dtype=float
    )
    ctx_scale = np.where(ctx_scale < 1e-12, 1.0, ctx_scale)

    context_rows, used = [], set()
    for q in PROBE_CONTEXT_QUANTILES:
        qvec  = ctx_data.quantile(q).to_numpy(dtype=float)
        dists = ((ctx_array - qvec) / ctx_scale) ** 2
        order = np.argsort(dists.sum(axis=1))
        pos   = next((int(p) for p in order if int(p) not in used), int(order[0]))
        used.add(pos)
        row = {"context_quantile": float(q)}
        row.update({f: float(ctx_data.iloc[pos][f]) for f in CONTEXT_FEATURES})
        context_rows.append(row)
    contexts = pd.DataFrame(context_rows)

    i_low  = float(pooled_train[CURRENT_FEATURE].quantile(PROBE_CURRENT_Q_LOW))
    i_high = float(pooled_train[CURRENT_FEATURE].quantile(PROBE_CURRENT_Q_HIGH))
    if not (np.isfinite(i_low) and np.isfinite(i_high) and i_high > i_low):
        raise ValueError(f"Invalid probe current range: {i_low}, {i_high}")
    current_values = np.linspace(i_low, i_high, PROBE_CURRENT_POINTS)

    probe_rows = []
    for ctx_id, ctx in contexts.iterrows():
        for step, current in enumerate(current_values):
            row = {
                "context_id":       int(ctx_id),
                "context_quantile": float(ctx["context_quantile"]),
                "current_step":     int(step),
                CURRENT_FEATURE:    float(current),
            }
            row.update({f: float(ctx[f]) for f in CONTEXT_FEATURES})
            probe_rows.append(row)
    raw_probe = pd.DataFrame(probe_rows)

    meta_cols = ["context_id", "context_quantile", "current_step"]
    norm_probe = raw_probe[meta_cols].copy()
    for feature in FEATURES:
        norm_probe[feature] = (
            raw_probe[feature] - standardizer["X_mean"][feature]
        ) / standardizer["X_std"][feature]

    info = {
        "construction": (
            "Five empirical operating contexts selected from pooled training "
            "observations nearest to q10/q25/q50/q75/q90 non-current quantile "
            "vectors; current is swept independently within each context."
        ),
        "context_quantiles":           PROBE_CONTEXT_QUANTILES,
        "current_points_per_context":  PROBE_CURRENT_POINTS,
        "current_range_basis": (
            f"Pooled training q{PROBE_CURRENT_Q_LOW:.2f} to "
            f"q{PROBE_CURRENT_Q_HIGH:.2f}"
        ),
        "current_min_A":  i_low,
        "current_max_A":  i_high,
        "n_contexts":     int(len(contexts)),
        "n_probe_points": int(len(raw_probe)),
    }
    return raw_probe, norm_probe, info


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    dirs = ensure_dirs(project_dir)

    print("=" * 72)
    print("NLR PEM PREPROCESSING")
    print(f"Project : {project_dir}")
    print(f"Raw dir : {dirs['raw']}")
    print("=" * 72)

    missing = [
        fn for fn in CLIENT_SOURCES.values()
        if not (dirs["raw"] / fn).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Place these files in data/nlr_raw:\n  - " + "\n  - ".join(missing)
        )

    all_client_splits: Dict[int, Dict[str, pd.DataFrame]] = {}
    manifest = {
        "protocol": {
            "lag_seconds":              LAG_SECONDS,
            "average_seconds":          AVERAGE_SECONDS,
            "min_valid_per_window":     MIN_VALID_PER_WINDOW,
            "split_fractions":          {
                "train": FRAC_TRAIN, "calib": FRAC_CALIB,
                "audit": FRAC_AUDIT, "test":  FRAC_TEST,
            },
            "embargo_seconds_each_boundary": EMBARGO_SECONDS,
            "standardizer_fit": "Client 1 train only",
            "feature_map":      FEATURE_MAP,
            "target":           TARGET,
        },
        "clients": {},
    }

    # ------------------------------------------------------------------
    # 1. Temporal preprocessing + chronological splitting
    # ------------------------------------------------------------------
    print("\n[1/5] 3-s alignment + 10-s averaging + chronological splitting")

    for client_id, filename in CLIENT_SOURCES.items():
        processed, exp_info, client_info = preprocess_client(
            dirs["raw"] / filename, client_id
        )

        processed.to_csv(
            dirs["clients"] / (
                f"client{client_id}_{CLIENT_LABELS[client_id]}_aligned10s.csv"
            ),
            index=False,
        )

        splits, split_details = chronological_split_client(processed)
        all_client_splits[client_id] = splits

        for part, df in splits.items():
            df.to_csv(
                dirs["splits_raw"] / f"client{client_id}_{part}.csv",
                index=False,
            )

        client_info["experiments"]  = exp_info
        client_info["split_details"] = split_details
        client_info["split_counts"] = {p: int(len(d)) for p, d in splits.items()}
        manifest["clients"][str(client_id)] = client_info

        print(
            f"\n  Client {client_id}: {CLIENT_LABELS[client_id]}"
            f"\n    10-s rows : {len(processed)}"
            f"\n    train/calib/audit/test : "
            f"{len(splits['train'])} / {len(splits['calib'])} / "
            f"{len(splits['audit'])} / {len(splits['test'])}"
        )

    # ------------------------------------------------------------------
    # 2. Fit Client-1 standardizer
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting standardizer on Client 1 TRAIN only")
    standardizer = fit_standardizer(all_client_splits[1]["train"])
    save_json(
        standardizer,
        dirs["metadata"] / "standardizer_client1_train.json",
    )
    print(
        f"  I_ref = {standardizer['I_mean']:.6f} A"
        f"\n  H_ref = {standardizer['y_mean']:.6f} kg/h"
        f"\n  y_std = {standardizer['y_std']:.6f} kg/h"
    )

    # ------------------------------------------------------------------
    # 3. Normalize all partitions
    # ------------------------------------------------------------------
    print("\n[3/5] Saving normalized splits")
    for client_id, splits in all_client_splits.items():
        for part, df in splits.items():
            normalize_split(df, standardizer).to_csv(
                dirs["splits_norm"] / f"client{client_id}_{part}_normalized.csv",
                index=False,
            )

    # ------------------------------------------------------------------
    # 4. Physics probe
    # ------------------------------------------------------------------
    print("\n[4/5] Building multi-context physics probe")
    pooled_train = pd.concat(
        [all_client_splits[cid]["train"] for cid in sorted(all_client_splits)],
        ignore_index=True,
    )
    raw_probe, norm_probe, probe_info = build_multicontext_probe(
        pooled_train, standardizer
    )
    raw_probe.to_csv(dirs["metadata"] / "physics_probe_raw.csv", index=False)
    norm_probe.to_csv(dirs["metadata"] / "physics_probe_normalized.csv", index=False)
    save_json(probe_info, dirs["metadata"] / "physics_probe_manifest.json")

    manifest["standardizer"]  = standardizer
    manifest["physics_probe"] = probe_info
    save_json(manifest, dirs["metadata"] / "nlr_preprocessing_manifest.json")

    print(
        f"  Probe = {probe_info['n_contexts']} contexts "
        f"x {probe_info['current_points_per_context']} current values "
        f"= {probe_info['n_probe_points']} points"
        f"\n  Current sweep = {probe_info['current_min_A']:.2f}"
        f" -> {probe_info['current_max_A']:.2f} A"
    )

    # ------------------------------------------------------------------
    # 5. Integrity checks
    # ------------------------------------------------------------------
    print("\n[5/5] Integrity checks")
    for client_id, splits in all_client_splits.items():
        for part, df in splits.items():
            vals = df[FEATURES + [TARGET]].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise AssertionError(
                    f"Non-finite values in Client {client_id} {part}."
                )

    expected = len(PROBE_CONTEXT_QUANTILES) * PROBE_CURRENT_POINTS
    if len(raw_probe) != expected:
        raise AssertionError("Unexpected physics probe size.")

    print("  PASS: no NaN/non-finite values in model-facing splits.")
    print("  PASS: physics probe size correct.")
    print("\nPREPROCESSING COMPLETE")
    print(f"\nProcessed data: {dirs['processed']}")
    print(f"Future NLR results: {dirs['results']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLR PEM preprocessing for PI-SDFL.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PI-SDFL project root (auto-detected when script is in <root>/src_nlr).",
    )
    args = parser.parse_args()
    main(args.project_dir)
