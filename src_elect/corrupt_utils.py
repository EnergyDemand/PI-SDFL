"""
corrupt_utils.py — Shared corruption utilities for current-injection experiments.

Imported by all run_corrupt_current_*.py files.
Do not run this file directly.

Corruption target: Client 4 (client4_data.csv, index 3).
Only y_train of Client 4 is modified.
Calib, audit, test, and global eval sets are never touched.

Default parameters (matching verified paper results):
    quantile = 0.80   top 20% of training rows by current
    frac     = 1.00   full inversion of current effect
"""

import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_utils import ClientSplit

CORRUPTED_CLIENT_INDEX  = 3
CORRUPTED_CLIENT_NUMBER = 4
CORRUPTED_CLIENT_FILE   = "client4_data.csv"

DEFAULT_QUANTILE = 0.80
DEFAULT_FRAC     = 1.00

OUT_DIR_NAME = "corrupt_current"


def _estimate_partial_beta_I(
    I_values: np.ndarray,
    T_values: np.ndarray,
    P_values: np.ndarray,
    y_values: np.ndarray,
    center:   float,
) -> Tuple[Optional[float], int]:
    """
    Estimate partial current coefficient beta_I via OLS controlling for T and P.
        y ~ beta_I*(I - center) + beta_T*T + beta_P*P + intercept

    Centering at threshold improves numerical conditioning without
    changing beta_I. Returns (beta_I, rank).
    """
    A = np.column_stack([
        I_values - center,
        T_values,
        P_values,
        np.ones(len(I_values), dtype=np.float64),
    ])
    coef, _, rank, _ = np.linalg.lstsq(A, y_values, rcond=None)
    return (float(coef[0]) if rank == 4 else None, int(rank))


def corrupt_split_current(
    cd:       ClientSplit,
    frac:     float = DEFAULT_FRAC,
    quantile: float = DEFAULT_QUANTILE,
) -> Tuple[ClientSplit, Dict]:
    """
    Current-component (partial-effect) corruption on y_train only.

    Selects rows in the top (1-quantile) fraction by current value.
    Estimates partial beta_I controlling for T and P via OLS on the
    full training split. For selected rows applies:
        y_new = y - 2*frac*beta_I*(I - threshold)

    Corruption is zero at threshold so the target field stays continuous.

    frac semantics (theoretical current effect after corruption):
        0.00  -> +beta_I       (no change)
        0.25  -> +0.5*beta_I   (weakened)
        0.50  ->  0            (flat)
        0.75  -> -0.5*beta_I   (partially inverted)
        1.00  -> -beta_I       (fully inverted)

    Only y_train is modified. Calib/audit/test are untouched.

    Raises ValueError if:
        - fewer than 10 rows selected
        - regression matrix rank-deficient
        - beta_I non-finite or <= 0
        - corrupted targets contain NaN/Inf
        - any corrupted H2 target is non-positive
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0,1], got {frac}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0,1), got {quantile}")

    I64 = np.asarray(cd.I_train,       dtype=np.float64)
    T64 = np.asarray(cd.X_train[:, 1], dtype=np.float64)
    P64 = np.asarray(cd.X_train[:, 2], dtype=np.float64)
    y64 = np.asarray(cd.y_train,       dtype=np.float64)

    threshold   = float(np.quantile(I64, quantile))
    high_mask   = I64 >= threshold
    n_corrupted = int(high_mask.sum())

    if n_corrupted < 10:
        raise ValueError(
            f"corrupt_split_current: only {n_corrupted} rows at "
            f"quantile={quantile:.2f} (threshold={threshold:.2f}A). "
            f"Need >= 10. Lower the quantile.")

    beta_I, rank = _estimate_partial_beta_I(I64, T64, P64, y64, threshold)

    if rank < 4:
        raise ValueError(
            f"corrupt_split_current: rank-deficient matrix (rank={rank}).")
    if beta_I is None or not np.isfinite(beta_I):
        raise ValueError(
            "corrupt_split_current: beta_I is non-finite.")
    if beta_I <= 0:
        raise ValueError(
            f"corrupt_split_current: beta_I={beta_I:.4f} <= 0 after "
            f"controlling for T,P. Cannot construct physics-violating corruption.")

    I_high        = I64[high_mask]
    y_high_before = y64[high_mask]
    y_new         = y64.copy()
    y_new[high_mask] = y_high_before - 2.0 * frac * beta_I * (I_high - threshold)

    if not np.all(np.isfinite(y_new)):
        raise ValueError(
            "corrupt_split_current: corrupted targets contain NaN/Inf.")
    if np.any(y_new[high_mask] <= 0):
        raise ValueError(
            "corrupt_split_current: non-positive H2 targets after corruption. "
            "Increase quantile or reduce frac.")

    theoretical_beta_after = (1.0 - 2.0 * frac) * beta_I

    beta_high_before, rank_high_before = _estimate_partial_beta_I(
        I64[high_mask], T64[high_mask], P64[high_mask], y_high_before, threshold)
    beta_high_after, rank_high_after = _estimate_partial_beta_I(
        I64[high_mask], T64[high_mask], P64[high_mask], y_new[high_mask], threshold)

    cd2 = copy.copy(cd)
    cd2.y_train = y_new.astype(cd.y_train.dtype, copy=False)

    diagnostics = {
        "corruption_type":                 "current",
        "corrupted_client_index":          CORRUPTED_CLIENT_INDEX,
        "corrupted_client_number":         CORRUPTED_CLIENT_NUMBER,
        "corrupted_client_file":           CORRUPTED_CLIENT_FILE,
        "current_quantile":                quantile,
        "current_threshold_A":             float(threshold),
        "severity":                        frac,
        "n_train":                         int(len(y64)),
        "n_corrupted":                     n_corrupted,
        "fraction_corrupted":              round(n_corrupted / len(y64), 4),
        "partial_beta_I":                  beta_I,
        "theoretical_partial_beta_after":  theoretical_beta_after,
        "regression_rank":                 int(rank),
        "high_region_partial_beta_before": beta_high_before,
        "high_region_partial_beta_after":  beta_high_after,
        "high_region_rank_before":         int(rank_high_before),
        "high_region_rank_after":          int(rank_high_after),
        "target_min_before":               float(y64.min()),
        "target_max_before":               float(y64.max()),
        "target_min_after":                float(y_new.min()),
        "target_max_after":                float(y_new.max()),
        "max_absolute_target_change":      float(np.max(np.abs(y_new - y64))),
        "calib_unchanged":                 True,
        "audit_unchanged":                 True,
        "test_unchanged":                  True,
    }
    return cd2, diagnostics


def apply_corruption(
    splits:   List[ClientSplit],
    frac:     float = DEFAULT_FRAC,
    quantile: float = DEFAULT_QUANTILE,
) -> Tuple[List[ClientSplit], Dict]:
    """
    Apply current-injection corruption to Client 4 only.
    Returns (corrupted_splits_list, diagnostics_dict).
    Clients 1-3 are unchanged references to the original splits.
    """
    cd_corrupt, diag = corrupt_split_current(
        splits[CORRUPTED_CLIENT_INDEX], frac=frac, quantile=quantile)
    corrupted_splits = (
        splits[:CORRUPTED_CLIENT_INDEX]
        + [cd_corrupt]
        + splits[CORRUPTED_CLIENT_INDEX + 1:]
    )
    return corrupted_splits, diag


def print_corruption_summary(diag: Dict) -> None:
    print(
        f"  [CORRUPT] {diag['corrupted_client_file']} y_train "
        f"current-injection q={diag['current_quantile']:.2f} "
        f"frac={diag['severity']:.2f}  "
        f"n_corrupted={diag['n_corrupted']} "
        f"({diag['fraction_corrupted']*100:.1f}%)  "
        f"beta_I: {diag['partial_beta_I']:.4f} -> "
        f"theoretical {diag['theoretical_partial_beta_after']:.4f}  "
        f"max|dy|={diag['max_absolute_target_change']:.1f}  "
        f"y_min_after={diag['target_min_after']:.2f}"
    )


def make_corrupt_config(mode: str) -> Tuple[float, float]:
    """
    Return (quantile, frac) for each mode.
    Final and pilot use paper-verified parameters.
    Smoke uses same params but fewer seeds/rounds (controlled by ExperimentConfig).
    """
    return DEFAULT_QUANTILE, DEFAULT_FRAC


def corrupt_out_dir(results_dir: str, quantile: float, frac: float) -> str:
    q_tag    = f"q{int(round(quantile * 100)):03d}"
    frac_tag = f"f{int(round(frac * 100)):03d}"
    return os.path.join(results_dir, f"{OUT_DIR_NAME}_{q_tag}_{frac_tag}")
