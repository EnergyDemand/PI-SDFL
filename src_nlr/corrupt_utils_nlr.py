"""
corrupt_utils_nlr.py — Input-space sensor fault for NLR experiments.

NLR corruption design (locked):
  - Fault type: gain error on Client 4 current INPUT only.
  - I_fault = gain * I_train  (gain = 1.1 for primary experiment)
  - The X_train[:, 0] column is multiplied by gain.
  - The normalized current column is recomputed from physical values.
  - H2 target (y_train) is NEVER modified.
  - Calib, audit, and test splits are NEVER modified.

This is fundamentally different from corrupt_utils.py (synthetic PEM),
which modifies y_train. Here the input is corrupted; the target is real.

Terminology (locked for AEI paper):
  "input-space sensor fault" — NOT "target corruption".
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from experiment_utils_nlr import ClientSplitNLR

CORRUPTED_CLIENT_INDEX  = 3    # 0-indexed position in splits list
CORRUPTED_CLIENT_NUMBER = 4    # 1-indexed client number
CORRUPTED_CLIENT_LABEL  = "simulated_tidal"

PRIMARY_GAIN = 1.1


def corrupt_split_sensor_fault(
    cd:   ClientSplitNLR,
    gain: float,
    std,
) -> Tuple[ClientSplitNLR, Dict]:
    """
    Apply a gain error to the current column of X_train only.

    The normalized current column (col 0) is:
        1. Denormalized back to physical Amperes.
        2. Multiplied by gain.
        3. Re-normalized using the same standardizer.

    All other columns and all other splits are unchanged.
    y_train is unchanged.

    Args:
        cd:   The ClientSplitNLR to corrupt (must be Client 4).
        gain: Multiplicative gain on physical current (e.g. 1.1 = +10%).
        std:  The fitted Standardizer (for denorm / renorm).

    Returns:
        (corrupted_split, diagnostics_dict)
    """
    if not (gain > 0):
        raise ValueError(f"gain must be > 0, got {gain}")

    I_phys_original = cd.I_train.copy()
    I_phys_faulted  = (I_phys_original * gain).astype(np.float32)

    I_norm_original = cd.X_train[:, 0].copy()
    I_norm_faulted  = (
        (I_phys_faulted - std.I_mean) / std.I_std
    ).astype(np.float32)

    X_train_faulted       = cd.X_train.copy()
    X_train_faulted[:, 0] = I_norm_faulted

    cd2          = copy.copy(cd)
    cd2.X_train  = X_train_faulted
    cd2.I_train  = I_phys_faulted

    diagnostics = {
        "corruption_type":              "input_sensor_fault",
        "fault_subtype":                "gain_error",
        "corrupted_client_index":       CORRUPTED_CLIENT_INDEX,
        "corrupted_client_number":      CORRUPTED_CLIENT_NUMBER,
        "corrupted_client_label":       CORRUPTED_CLIENT_LABEL,
        "gain":                         float(gain),
        "n_train_rows_affected":        int(len(I_phys_original)),
        "I_phys_mean_before":           float(I_phys_original.mean()),
        "I_phys_mean_after":            float(I_phys_faulted.mean()),
        "I_phys_min_before":            float(I_phys_original.min()),
        "I_phys_max_before":            float(I_phys_original.max()),
        "I_phys_min_after":             float(I_phys_faulted.min()),
        "I_phys_max_after":             float(I_phys_faulted.max()),
        "I_norm_mean_before":           float(I_norm_original.mean()),
        "I_norm_mean_after":            float(I_norm_faulted.mean()),
        "target_modified":              False,
        "calib_unchanged":              True,
        "audit_unchanged":              True,
        "test_unchanged":               True,
    }
    return cd2, diagnostics


def apply_sensor_fault(
    splits: List[ClientSplitNLR],
    gain:   float,
    std,
) -> Tuple[List[ClientSplitNLR], Dict]:
    """
    Apply the gain-error sensor fault to Client 4 only.

    Args:
        splits: List of 4 ClientSplitNLR objects (indices 0-3).
        gain:   Multiplicative gain on physical current.
        std:    Fitted Standardizer.

    Returns:
        (corrupted_splits_list, diagnostics_dict)
    """
    cd_corrupt, diag = corrupt_split_sensor_fault(
        splits[CORRUPTED_CLIENT_INDEX], gain=gain, std=std)
    corrupted = (
        splits[:CORRUPTED_CLIENT_INDEX]
        + [cd_corrupt]
        + splits[CORRUPTED_CLIENT_INDEX + 1:]
    )
    return corrupted, diag


def print_fault_summary(diag: Dict) -> None:
    print(
        f"  [SENSOR FAULT] Client {diag['corrupted_client_number']} "
        f"({diag['corrupted_client_label']}) "
        f"gain={diag['gain']:.2f}  "
        f"n_affected={diag['n_train_rows_affected']}  "
        f"I_phys_mean: {diag['I_phys_mean_before']:.1f} -> "
        f"{diag['I_phys_mean_after']:.1f} A  "
        f"target_modified={diag['target_modified']}"
    )
