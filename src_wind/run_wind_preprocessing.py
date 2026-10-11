"""
run_wind_preprocessing.py
=========================
AEI wind preprocessing pipeline for PI-SDFL.

Protocol
--------
Four clients (fixed ring order):
    Client 0: WT3 — initial/warm-start client (largest, most varied)
    Client 1: WT2
    Client 2: WT4
    Client 3: WT1 (smallest)

Input features (raw column -> model-facing name):
    V           -> wind_speed    [m/s]   physics feature: dP/dV >= 0
    D           -> wind_dir      [deg]   directional — held fixed in probe
    air density -> air_density   [kg/m³]
    I           -> turbulence    [dimensionless]
    S_b         -> wind_shear    [exponent]      — held fixed in probe

Target:
    y (% relative to rated power) -> power_pct [%]

Power filter:
    1% < power_pct < 95%  (below-rated constraint-valid operating region)

Chronological split PER TURBINE:
    65% train / 10% calibration / 10% audit / 15% test

    A 200-record embargo is inserted at each of the three boundaries
    (train->calib, calib->audit, audit->test).

    Justification: lag-1 autocorrelation ≈ 0.95 in all turbines;
    autocorrelation decays to < 0.06 by lag-200 in all turbines.

Standardizer:
    Fitted ONLY on Client 0 (WT3) training data.

Physics probe:
    Multi-context empirical probe from POOLED training data.
    Five operating contexts at quantiles q10/q25/q50/q75/q90 of the
    non-wind-speed feature set (air_density, turbulence).
    wind_dir and wind_shear are held at their training medians
    (directional / sign-uncertain — not swept).
    wind_speed is swept over 50 points within each context.
    Total: 5 × 50 = 250 probe points.

    Note: the probe uses training data only. wind_dir and wind_shear
    are frozen at their global training medians across all contexts,
    not at per-context quantiles, because sweeping circular or
    sign-uncertain features does not test the monotonicity physics.

Physics law:
    Below-rated wind-speed–power monotonicity prior: dP/dV >= 0.
    NOT claimed to be strict Betz-law enforcement.

Global evaluation:
    global_test.csv = concatenated per-client test splits (same as NLR).
    No random global_val/global_test — evaluation is fully leakage-safe.

Usage
-----
Place raw turbine CSV files in:
    data\\wind_raw\\

Files expected:
    Inland_Wind_Farm_Dataset1_WT1_.csv
    Inland_Wind_Farm_Dataset1_WT2_.csv
    Inland_Wind_Farm_Dataset1_WT3_.csv
    Inland_Wind_Farm_Dataset1_WT4_.csv

Then run:
    cd /d D:\\GPU\\pi_sdfl_v2
    python src_wind_aei\\run_wind_preprocessing.py
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

# Fixed ring order: WT3 -> WT2 -> WT4 -> WT1
CLIENT_SOURCES = {
    0: "Inland_Wind_Farm_Dataset1_WT3_.csv",
    1: "Inland_Wind_Farm_Dataset1_WT2_.csv",
    2: "Inland_Wind_Farm_Dataset1_WT4_.csv",
    3: "Inland_Wind_Farm_Dataset1_WT1_.csv",
}

CLIENT_LABELS = {
    0: "WT3_initial",
    1: "WT2",
    2: "WT4",
    3: "WT1",
}

# Raw column -> model-facing feature name
FEATURE_MAP = {
    "V":           "wind_speed",
    "D":           "wind_dir",
    "air density": "air_density",
    "I":           "turbulence",
    "S_b":         "wind_shear",
}

FEATURES         = list(FEATURE_MAP.values())   # ['wind_speed','wind_dir','air_density','turbulence','wind_shear']
PHYSICS_FEATURE  = "wind_speed"                 # index 0 — dP/dV >= 0
TARGET_RAW_COL   = "y (% relative to rated power)"
TARGET           = "power_pct"
SEQ_COL          = "Sequence No."

# Features swept/frozen in the probe grid
# wind_speed: swept (physics feature)
# air_density, turbulence: context quantiles
# wind_dir, wind_shear: frozen at training median (circular / sign-uncertain)
PROBE_CONTEXT_FEATURES = ["air_density", "turbulence"]
PROBE_FROZEN_FEATURES  = ["wind_dir", "wind_shear"]

# Power filter: below-rated constraint-valid operating region
POWER_LO = 1.0    # % rated
POWER_HI = 95.0   # % rated

# Chronological split fractions
FRAC_TRAIN = 0.65
FRAC_CALIB = 0.10
FRAC_AUDIT = 0.10
FRAC_TEST  = 0.15

# Embargo: autocorrelation < 0.06 by lag-200 in all turbines
EMBARGO_RECORDS = 200

# Physics probe
PROBE_WIND_POINTS       = 50
PROBE_CONTEXT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
PROBE_WIND_Q_LOW        = 0.01
PROBE_WIND_Q_HIGH       = 0.99


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
        "raw":         project_dir / "data" / "wind_raw",
        "processed":   project_dir / "data" / "wind_processed",
        "clients":     project_dir / "data" / "wind_processed" / "clients_chronological",
        "splits_raw":  project_dir / "data" / "wind_processed" / "splits_raw",
        "splits_norm": project_dir / "data" / "wind_processed" / "splits_normalized",
        "metadata":    project_dir / "data" / "wind_processed" / "metadata",
        "results":     project_dir / "results" / "wind",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, source_name: str) -> None:
    missing_feat = [c for c in FEATURE_MAP if c not in df.columns]
    missing_tgt  = [] if TARGET_RAW_COL in df.columns else [TARGET_RAW_COL]
    missing_seq  = [] if SEQ_COL in df.columns else [SEQ_COL]
    missing = missing_feat + missing_tgt + missing_seq
    if missing:
        raise KeyError(f"{source_name}: missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Power filter + chronological split
# ---------------------------------------------------------------------------

def filter_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply power filter and rename columns to model-facing names.
    Preserves chronological order (Sequence No. order).
    """
    mask = (df[TARGET_RAW_COL] > POWER_LO) & (df[TARGET_RAW_COL] < POWER_HI)
    df   = df[mask].copy()
    df   = df.sort_values(SEQ_COL).reset_index(drop=True)

    rename = {raw: model for raw, model in FEATURE_MAP.items()}
    rename[TARGET_RAW_COL] = TARGET
    df = df.rename(columns=rename)

    df["seq_no"] = df[SEQ_COL].values
    keep = ["seq_no"] + FEATURES + [TARGET]
    return df[keep].reset_index(drop=True)


def chronological_split(
    df:      pd.DataFrame,
    client_id: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    65/10/10/15 chronological partition with 200-record embargo at each boundary.

    Fractions applied AFTER reserving the three embargo blocks.
    """
    n         = len(df)
    total_gap = 3 * EMBARGO_RECORDS

    if n <= total_gap + 40:
        raise ValueError(
            f"Client {client_id}: only {n} filtered records — "
            "too short for the requested split + embargo protocol."
        )

    usable  = n - total_gap
    n_train = int(np.floor(FRAC_TRAIN * usable))
    n_calib = int(np.floor(FRAC_CALIB * usable))
    n_audit = int(np.floor(FRAC_AUDIT * usable))
    n_test  = usable - n_train - n_calib - n_audit

    i = 0
    train = df.iloc[i : i + n_train].copy();           i += n_train + EMBARGO_RECORDS
    calib = df.iloc[i : i + n_calib].copy();           i += n_calib + EMBARGO_RECORDS
    audit = df.iloc[i : i + n_audit].copy();           i += n_audit + EMBARGO_RECORDS
    test  = df.iloc[i : i + n_test ].copy()

    splits = {"train": train, "calib": calib, "audit": audit, "test": test}

    info = {
        "client_id":                  client_id,
        "client_label":               CLIENT_LABELS[client_id],
        "filtered_rows":              int(n),
        "embargo_records_per_boundary": EMBARGO_RECORDS,
        "usable_rows":                int(usable),
        "train":                      int(n_train),
        "calib":                      int(n_calib),
        "audit":                      int(n_audit),
        "test":                       int(n_test),
        "train_seq_range":            [int(train["seq_no"].min()), int(train["seq_no"].max())],
        "test_seq_range":             [int(test["seq_no"].min()),  int(test["seq_no"].max())],
    }
    return splits, info


# ---------------------------------------------------------------------------
# Standardizer (Client 0 training only)
# ---------------------------------------------------------------------------

def fit_standardizer(client0_train: pd.DataFrame) -> Dict:
    """
    Fit on Client 0 (WT3) training data only.
    Population std (ddof=0) — consistent with PI-SDFL implementation.
    """
    X = client0_train[FEATURES].to_numpy(dtype=np.float64)
    y = client0_train[TARGET].to_numpy(dtype=np.float64)

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0, ddof=0)
    y_mean = float(y.mean())
    y_std  = float(y.std(ddof=0))

    bad = [FEATURES[i] for i, s in enumerate(X_std) if s < 1e-12]
    if bad:
        raise ValueError(f"Near-zero training std for features: {bad}")
    if y_std < 1e-12:
        raise ValueError("Near-zero target std.")
    if X_mean[0] <= 0:
        raise ValueError(
            f"wind_speed mean = {X_mean[0]:.4f} must be > 0 for the hard-guard architecture.")
    if y_mean <= 0:
        raise ValueError(
            f"power_pct mean = {y_mean:.4f} must be > 0 for the P_ref scaling.")

    phys_idx = FEATURES.index(PHYSICS_FEATURE)
    return {
        "features":   FEATURES,
        "target":     TARGET,
        "X_mean":     {f: float(v) for f, v in zip(FEATURES, X_mean)},
        "X_std":      {f: float(v) for f, v in zip(FEATURES, X_std)},
        "y_mean":     y_mean,
        "y_std":      y_std,
        "PHYS_mean":  float(X_mean[phys_idx]),
        "PHYS_std":   float(X_std[phys_idx]),
        "fit_source": "Client 0 (WT3) training split only",
        "ddof":       0,
    }


def normalize_split(df: pd.DataFrame, standardizer: Dict) -> pd.DataFrame:
    out = df[["seq_no"]].copy()
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
    pooled_train: pd.DataFrame,
    standardizer: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Empirical physics probe — same design as NLR probe.

    Five operating contexts from POOLED training data nearest to the
    q10/q25/q50/q75/q90 quantile vectors of [air_density, turbulence].

    wind_dir and wind_shear are frozen at their global training medians
    across ALL contexts (circular / sign-uncertain — not swept).

    wind_speed is swept independently within each context over 50 points
    spanning the pooled training q01–q99 range.

    Total probe points: 5 contexts × 50 wind_speed values = 250.
    """
    ctx_data  = pooled_train[PROBE_CONTEXT_FEATURES].astype(float)
    ctx_array = ctx_data.to_numpy(dtype=float)

    ctx_scale = np.array(
        [standardizer["X_std"][f] for f in PROBE_CONTEXT_FEATURES], dtype=float)
    ctx_scale = np.where(ctx_scale < 1e-12, 1.0, ctx_scale)

    # Frozen values for wind_dir and wind_shear (training medians)
    frozen_vals = {
        f: float(pooled_train[f].median())
        for f in PROBE_FROZEN_FEATURES
    }

    context_rows, used = [], set()
    for q in PROBE_CONTEXT_QUANTILES:
        qvec  = ctx_data.quantile(q).to_numpy(dtype=float)
        dists = ((ctx_array - qvec) / ctx_scale) ** 2
        order = np.argsort(dists.sum(axis=1))
        pos   = next((int(p) for p in order if int(p) not in used), int(order[0]))
        used.add(pos)
        row = {"context_quantile": float(q)}
        row.update({f: float(ctx_data.iloc[pos][f]) for f in PROBE_CONTEXT_FEATURES})
        row.update(frozen_vals)
        context_rows.append(row)
    contexts = pd.DataFrame(context_rows)

    # wind_speed sweep range: pooled training q01–q99
    v_low  = float(pooled_train[PHYSICS_FEATURE].quantile(PROBE_WIND_Q_LOW))
    v_high = float(pooled_train[PHYSICS_FEATURE].quantile(PROBE_WIND_Q_HIGH))
    if not (np.isfinite(v_low) and np.isfinite(v_high) and v_high > v_low):
        raise ValueError(f"Invalid probe wind_speed range: {v_low}, {v_high}")
    wind_values = np.linspace(v_low, v_high, PROBE_WIND_POINTS)

    # Build raw probe: one row per (context, wind_speed)
    probe_rows = []
    for ctx_id, ctx in contexts.iterrows():
        for step, wind in enumerate(wind_values):
            row = {
                "context_id":       int(ctx_id),
                "context_quantile": float(ctx["context_quantile"]),
                "wind_step":        int(step),
                PHYSICS_FEATURE:    float(wind),
            }
            for f in PROBE_CONTEXT_FEATURES:
                row[f] = float(ctx[f])
            for f in PROBE_FROZEN_FEATURES:
                row[f] = float(ctx[f])
            probe_rows.append(row)
    raw_probe = pd.DataFrame(probe_rows)

    # Normalized probe: same meta cols, all 5 features normalized
    meta_cols = ["context_id", "context_quantile", "wind_step"]
    norm_probe = raw_probe[meta_cols].copy()
    for feature in FEATURES:
        norm_probe[feature] = (
            raw_probe[feature] - standardizer["X_mean"][feature]
        ) / standardizer["X_std"][feature]

    info = {
        "construction": (
            "Five empirical operating contexts from pooled training data "
            "nearest to q10/q25/q50/q75/q90 of [air_density, turbulence] quantile vectors. "
            "wind_dir and wind_shear frozen at global training medians. "
            "wind_speed swept independently within each context."
        ),
        "context_quantiles":          PROBE_CONTEXT_QUANTILES,
        "probe_context_features":     PROBE_CONTEXT_FEATURES,
        "probe_frozen_features":      PROBE_FROZEN_FEATURES,
        "frozen_feature_values":      frozen_vals,
        "wind_points_per_context":    PROBE_WIND_POINTS,
        "wind_range_basis":           f"Pooled training q{PROBE_WIND_Q_LOW:.2f} to q{PROBE_WIND_Q_HIGH:.2f}",
        "wind_speed_min_ms":          v_low,
        "wind_speed_max_ms":          v_high,
        "n_contexts":                 int(len(contexts)),
        "n_probe_points":             int(len(raw_probe)),
    }
    return raw_probe, norm_probe, info


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    dirs = ensure_dirs(project_dir)

    print("=" * 72)
    print("WIND AEI PREPROCESSING")
    print(f"Project : {project_dir}")
    print(f"Raw dir : {dirs['raw']}")
    print("=" * 72)

    missing = [fn for fn in CLIENT_SOURCES.values()
               if not (dirs["raw"] / fn).exists()]
    if missing:
        raise FileNotFoundError(
            "Place these files in data/wind_raw/:\n  - " + "\n  - ".join(missing))

    all_splits:  Dict[int, Dict[str, pd.DataFrame]] = {}
    split_infos: Dict[int, Dict]                    = {}

    manifest = {
        "protocol": {
            "power_filter":                {"gt": POWER_LO, "lt": POWER_HI, "unit": "% rated"},
            "split_fractions":             {"train": FRAC_TRAIN, "calib": FRAC_CALIB,
                                            "audit": FRAC_AUDIT, "test":  FRAC_TEST},
            "embargo_records_per_boundary": EMBARGO_RECORDS,
            "embargo_justification":       "lag-1 autocorrelation ~0.95; decays to <0.06 by lag-200",
            "chronological_order":         "Sequence No. ascending within each turbine",
            "standardizer_fit":            "Client 0 (WT3) training split only",
            "physics_law":                 "Below-rated wind-speed-power monotonicity prior: dP/dV >= 0",
            "physics_law_note":            "NOT strict Betz-law enforcement",
            "feature_map":                 FEATURE_MAP,
            "target_raw_col":              TARGET_RAW_COL,
            "target_model_col":            TARGET,
            "ring_order":                  "WT3 (C0) -> WT2 (C1) -> WT4 (C2) -> WT1 (C3)",
        },
        "clients": {},
    }

    # ------------------------------------------------------------------
    # 1. Filter, rename, chronological split
    # ------------------------------------------------------------------
    print("\n[1/5] Power filter + chronological split")

    for client_id, filename in CLIENT_SOURCES.items():
        source = dirs["raw"] / filename
        print(f"\n  Reading {filename}")

        df = pd.read_csv(source, low_memory=False)
        validate_columns(df, filename)

        raw_rows = len(df)
        df_clean = filter_and_rename(df)
        filtered_rows = len(df_clean)
        removed = raw_rows - filtered_rows

        print(f"    Raw rows     : {raw_rows}")
        print(f"    After filter : {filtered_rows} ({removed} removed, "
              f"{removed/raw_rows*100:.1f}%)")

        df_clean.to_csv(
            dirs["clients"] / f"client{client_id}_{CLIENT_LABELS[client_id]}_filtered.csv",
            index=False)

        splits, info = chronological_split(df_clean, client_id)
        all_splits[client_id]  = splits
        split_infos[client_id] = info

        for part, df_part in splits.items():
            df_part.to_csv(
                dirs["splits_raw"] / f"client{client_id}_{part}.csv",
                index=False)

        print(f"    train={info['train']}  calib={info['calib']}  "
              f"audit={info['audit']}  test={info['test']}")
        print(f"    train seq: {info['train_seq_range']}  "
              f"test seq: {info['test_seq_range']}")

        manifest["clients"][str(client_id)] = {
            "client_id":      client_id,
            "client_label":   CLIENT_LABELS[client_id],
            "source_file":    filename,
            "source_sha256":  sha256_short(source),
            "raw_rows":       int(raw_rows),
            "filtered_rows":  int(filtered_rows),
            **info,
        }

    # ------------------------------------------------------------------
    # 2. Fit Client 0 standardizer
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting standardizer on Client 0 (WT3) TRAIN only")
    standardizer = fit_standardizer(all_splits[0]["train"])
    save_json(standardizer,
              dirs["metadata"] / "standardizer_client0_train.json")
    print(f"  PHYS_mean (wind_speed) = {standardizer['PHYS_mean']:.4f} m/s")
    print(f"  y_mean    (power_pct)  = {standardizer['y_mean']:.4f} %")
    print(f"  y_std                  = {standardizer['y_std']:.4f} %")

    # ------------------------------------------------------------------
    # 3. Normalize all client partitions
    # ------------------------------------------------------------------
    print("\n[3/5] Saving normalized splits")
    for client_id, splits in all_splits.items():
        for part, df_part in splits.items():
            normalize_split(df_part, standardizer).to_csv(
                dirs["splits_norm"] / f"client{client_id}_{part}_normalized.csv",
                index=False)

    # ------------------------------------------------------------------
    # 4. Build global test set (concatenated client test splits)
    # ------------------------------------------------------------------
    print("\n[4/5] Building global test set")
    test_raw_parts  = [all_splits[cid]["test"] for cid in sorted(all_splits)]
    test_norm_parts = [
        pd.read_csv(dirs["splits_norm"] / f"client{cid}_test_normalized.csv")
        for cid in sorted(all_splits)
    ]
    global_test_raw  = pd.concat(test_raw_parts,  ignore_index=True)
    global_test_norm = pd.concat(test_norm_parts, ignore_index=True)
    global_test_raw.to_csv( dirs["splits_raw"]  / "global_test.csv", index=False)
    global_test_norm.to_csv(dirs["splits_norm"] / "global_test_normalized.csv", index=False)
    print(f"  global_test rows (raw)  : {len(global_test_raw)}")
    print(f"  global_test rows (norm) : {len(global_test_norm)}")

    # ------------------------------------------------------------------
    # 5. Multi-context physics probe
    # ------------------------------------------------------------------
    print("\n[5/6] Building multi-context physics probe")
    pooled_train = pd.concat(
        [all_splits[cid]["train"] for cid in sorted(all_splits)],
        ignore_index=True)
    raw_probe, norm_probe, probe_info = build_multicontext_probe(
        pooled_train, standardizer)

    raw_probe.to_csv( dirs["metadata"] / "physics_probe_raw.csv",        index=False)
    norm_probe.to_csv(dirs["metadata"] / "physics_probe_normalized.csv", index=False)
    save_json(probe_info, dirs["metadata"] / "physics_probe_manifest.json")

    print(f"  Probe = {probe_info['n_contexts']} contexts "
          f"x {probe_info['wind_points_per_context']} wind_speed values "
          f"= {probe_info['n_probe_points']} points")
    print(f"  wind_speed sweep = {probe_info['wind_speed_min_ms']:.3f} "
          f"-> {probe_info['wind_speed_max_ms']:.3f} m/s")
    print(f"  wind_dir  frozen = {probe_info['frozen_feature_values']['wind_dir']:.2f} deg")
    print(f"  wind_shear frozen= {probe_info['frozen_feature_values']['wind_shear']:.4f}")

    # ------------------------------------------------------------------
    # 6. Integrity checks
    # ------------------------------------------------------------------
    print("\n[6/6] Integrity checks")

    # No NaN/non-finite in any normalized split
    for client_id in sorted(all_splits):
        for part, df_part in all_splits[client_id].items():
            vals = df_part[FEATURES + [TARGET]].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise AssertionError(
                    f"Non-finite values in Client {client_id} {part}.")

    # Chronological order preserved in each split
    for client_id in sorted(all_splits):
        for part in ["train", "calib", "audit", "test"]:
            seq = all_splits[client_id][part]["seq_no"].values
            if not np.all(np.diff(seq) > 0):
                raise AssertionError(
                    f"Sequence order violated in Client {client_id} {part}.")

    # No sequence overlap between train and test per client
    for client_id in sorted(all_splits):
        tr_max = all_splits[client_id]["train"]["seq_no"].max()
        te_min = all_splits[client_id]["test"]["seq_no"].min()
        if te_min <= tr_max:
            raise AssertionError(
                f"Client {client_id}: test seq_no overlaps with train!")
        gap = te_min - tr_max - 1
        if gap < EMBARGO_RECORDS * 2:
            raise AssertionError(
                f"Client {client_id}: gap between train end and test start "
                f"= {gap} records, expected >= {EMBARGO_RECORDS * 2}.")

    # Probe size
    expected = len(PROBE_CONTEXT_QUANTILES) * PROBE_WIND_POINTS
    if len(raw_probe) != expected:
        raise AssertionError(f"Unexpected probe size: {len(raw_probe)} != {expected}.")

    # Monotone wind_speed within each context block
    for ctx_id, g in raw_probe.groupby("context_id"):
        diffs = g[PHYSICS_FEATURE].diff().dropna()
        if not (diffs > 0).all():
            raise AssertionError(
                f"wind_speed not strictly increasing in probe context {ctx_id}.")

    print("  PASS: no NaN/non-finite in model-facing splits.")
    print("  PASS: chronological order preserved in all splits.")
    print("  PASS: no sequence overlap between train and test.")
    print("  PASS: physics probe size correct.")
    print("  PASS: wind_speed monotone within each probe context.")

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------
    manifest["standardizer"]  = standardizer
    manifest["physics_probe"] = probe_info
    manifest["global_test"] = {
        "construction": "Concatenated per-client test splits (Client 0..3)",
        "raw_rows":     int(len(global_test_raw)),
    }
    save_json(manifest,
              dirs["metadata"] / "wind_aei_preprocessing_manifest.json")

    print("\nPREPROCESSING COMPLETE")
    print(f"\nProcessed data : {dirs['processed']}")
    print(f"Future results : {dirs['results']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wind AEI preprocessing for PI-SDFL.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PI-SDFL project root (auto-detected when script is in <root>/src_wind_aei).",
    )
    args = parser.parse_args()
    main(args.project_dir)
