"""
experiment_utils_nlr.py — Shared infrastructure for all NLR runner scripts.

Key differences from experiment_utils.py:
  - Loads pre-split, pre-normalized CSVs from data/nlr_processed/splits_normalized/
  - Loads pre-built physics probe from data/nlr_processed/metadata/
  - Loads standardizer from JSON (never re-fits it)
  - ClientSplitNLR holds splits in physical-unit y for evaluation
    and normalized X / normalized y for training.
  - Global evaluation set = concatenated per-client test splits.
  - Results go to results_nlr/ (separate from results_new/).

Do not run this file directly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nlr as C


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

NORM_DIR    = os.path.join(PROJECT_DIR, "data", "nlr_processed", "splits_normalized")
RAW_DIR     = os.path.join(PROJECT_DIR, "data", "nlr_processed", "splits_raw")
META_DIR    = os.path.join(PROJECT_DIR, "data", "nlr_processed", "metadata")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "nlr")

STANDARDIZER_JSON = os.path.join(META_DIR, "standardizer_client1_train.json")
PROBE_NORM_CSV    = os.path.join(META_DIR, "physics_probe_normalized.csv")
PROBE_RAW_CSV     = os.path.join(META_DIR, "physics_probe_raw.csv")

N_CLIENTS = 4
CLIENT_LABELS = [
    "historical_wind",
    "historical_solar_pv",
    "simulated_wind",
    "simulated_tidal",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm_path(client_id: int, part: str) -> str:
    return os.path.join(NORM_DIR, f"client{client_id}_{part}_normalized.csv")


def _raw_path(client_id: int, part: str) -> str:
    return os.path.join(RAW_DIR, f"client{client_id}_{part}.csv")


# ---------------------------------------------------------------------------
# ExperimentConfig — identical field names to experiment_utils.py
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    mode:             str   = "smoke"
    seeds:            tuple = (0,)
    rounds:           int   = 2
    pretrain_ep:      int   = 5
    local_epochs:     int   = 2
    calib_epochs:     int   = 5
    eval_head_ep:     int   = 5
    lr_body:          float = 1e-3
    lr_head:          float = 1e-3
    use_hard_guard:   bool  = True
    use_mono_loss:    bool  = True
    beta_mono:        float = 1.0
    use_anchor:       bool  = True
    mu_anchor_start:  float = 1.0
    mu_anchor_floor:  float = 0.3
    use_gate:         bool  = True
    tau_pvr:          float = C.TAU_PVR
    tau_mag:          float = C.TAU_MAG
    epsilon_r2:       float = C.EPSILON_R2
    cwc_lambda:       float = C.CWC_LAMBDA


def make_config(mode: str) -> ExperimentConfig:
    if mode == "smoke":
        return ExperimentConfig(
            mode="smoke", seeds=(0,),
            rounds=2, pretrain_ep=5, local_epochs=2,
            calib_epochs=5, eval_head_ep=5,
        )
    if mode == "pilot":
        return ExperimentConfig(
            mode="pilot", seeds=(0, 1, 2),
            rounds=10, pretrain_ep=50, local_epochs=20,
            calib_epochs=20, eval_head_ep=30,
        )
    if mode == "final":
        return ExperimentConfig(
            mode="final", seeds=tuple(range(10)),
            rounds=20, pretrain_ep=200, local_epochs=50,
            calib_epochs=50, eval_head_ep=50,
        )
    raise ValueError(f"Unknown mode '{mode}'. Use: smoke / pilot / final")


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode", type=str, default="smoke",
        choices=["smoke", "pilot", "final"],
    )
    return parser


# ---------------------------------------------------------------------------
# ClientSplitNLR
# ---------------------------------------------------------------------------

class ClientSplitNLR:
    """
    Holds one client's pre-split NLR data.

    Training uses normalized X and normalized y (from splits_normalized/).
    Evaluation uses normalized X and physical-unit y (denormalized from std).

    I_phys is extracted from the normalized X col 0, then denormalized back
    to physical Amperes — this is the raw current value used in the forward()
    call and by the gate/physics metrics.
    """

    __slots__ = (
        "X_train", "y_train", "I_train",
        "X_calib", "y_calib", "I_calib",
        "X_audit", "y_audit", "I_audit",
        "X_test",  "y_test",  "I_test",
        "n_train", "client_id", "label",
    )

    def __init__(
        self,
        client_id: int,
        std:       C.Standardizer,
    ) -> None:
        self.client_id = client_id
        self.label     = CLIENT_LABELS[client_id - 1]

        def _load(part: str):
            X_norm, y_norm = C.load_split_csv(_norm_path(client_id, part))
            # Denormalize y to physical units for evaluation.
            y_raw  = std.inverse_y(y_norm)
            # Denormalize current column to physical Amperes.
            I_phys = (X_norm[:, 0] * std.I_std + std.I_mean).astype(np.float32)
            return X_norm, y_raw, I_phys, y_norm

        X_tr, y_tr, I_tr, yn_tr = _load("train")
        X_ca, y_ca, I_ca, _     = _load("calib")
        X_au, y_au, I_au, _     = _load("audit")
        X_te, y_te, I_te, _     = _load("test")

        self.X_train, self.y_train, self.I_train = X_tr, yn_tr, I_tr
        self.X_calib, self.y_calib, self.I_calib = X_ca, y_ca,  I_ca
        self.X_audit, self.y_audit, self.I_audit = X_au, y_au,  I_au
        self.X_test,  self.y_test,  self.I_test  = X_te, y_te,  I_te
        self.n_train = len(X_tr)

    @property
    def y_train_norm(self) -> np.ndarray:
        return self.y_train

    @property
    def y_train_phys(self) -> np.ndarray:
        raise AttributeError(
            "y_train is stored normalized. Use std.inverse_y(split.y_train).")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Module-level eval mode — set by load_data(), read by per-client helpers.
# Ensures helpers use audit for smoke/pilot and test for final without
# requiring every runner call site to pass mode explicitly.
_EVAL_MODE: str = "final"


def load_data(
    cfg: ExperimentConfig,
) -> Tuple[
    List[ClientSplitNLR],
    C.Standardizer,
    np.ndarray,   # X_eval_norm   [N_eval_global, 6]
    np.ndarray,   # y_eval_raw    [N_eval_global]   physical kg/h
    np.ndarray,   # I_eval_phys   [N_eval_global]   physical A
]:
    global _EVAL_MODE
    _EVAL_MODE = cfg.mode

    std = C.Standardizer.from_json(STANDARDIZER_JSON)
    splits = [ClientSplitNLR(client_id=i, std=std) for i in range(1, N_CLIENTS + 1)]

    # smoke/pilot use audit partitions — final test remains untouched
    # until the protocol is frozen and final mode is run.
    if cfg.mode == "final":
        parts = [(sp.X_test, sp.y_test, sp.I_test) for sp in splits]
    else:
        parts = [(sp.X_audit, sp.y_audit, sp.I_audit) for sp in splits]

    X_eval_norm  = np.vstack([p[0] for p in parts])
    y_eval_raw   = np.concatenate([p[1] for p in parts])
    I_eval_phys  = np.concatenate([p[2] for p in parts])

    return splits, std, X_eval_norm, y_eval_raw, I_eval_phys


def load_probe(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return C.load_probe(PROBE_NORM_CSV, PROBE_RAW_CSV, device)


def build_split_manifest(
    cfg:    ExperimentConfig,
    splits: List[ClientSplitNLR],
) -> Dict:
    return {
        "split_source":    "pre-built chronological splits from run_nlr_preprocessing.py",
        "standardizer":    STANDARDIZER_JSON,
        "probe_norm":      PROBE_NORM_CSV,
        "probe_raw":       PROBE_RAW_CSV,
        "client_sizes":    {sp.label: sp.n_train for sp in splits},
        "eval_set": (
            "concatenated per-client test splits — final held-out evaluation"
            if cfg.mode == "final" else
            "concatenated per-client audit splits — development only"
        ),
        "data_file_hashes": {
            f"client{sp.client_id}_{part}": C.file_hash(
                _norm_path(sp.client_id, part))
            for sp in splits
            for part in ["train", "calib", "audit", "test"]
        },
    }


# ---------------------------------------------------------------------------
# Model helpers (identical API to experiment_utils.py)
# ---------------------------------------------------------------------------

def new_model(use_hard_guard: bool) -> C.ElectrolyzerMLP:
    return C.ElectrolyzerMLP(use_hard_guard=use_hard_guard).to(DEVICE)


def mu_schedule(r: int, cfg: ExperimentConfig) -> float:
    if not cfg.use_anchor:
        return 0.0
    frac = r / max(cfg.rounds - 1, 1)
    return cfg.mu_anchor_floor + (cfg.mu_anchor_start - cfg.mu_anchor_floor) * (1.0 - frac)


def recalib_head(
    model:  C.ElectrolyzerMLP,
    cd:     ClientSplitNLR,
    std:    C.Standardizer,
    cfg:    ExperimentConfig,
    seed:   int,
    use_pi: bool,
) -> None:
    C.train_local(
        model=model, X_norm=cd.X_calib,
        y_norm=std.transform_y(cd.y_calib), I_phys=cd.I_calib,
        n_epochs=cfg.eval_head_ep, lr_body=0.0, lr_head=cfg.lr_head,
        seed=seed, use_mono_loss=False, beta_mono=0.0,
        use_hard_guard=use_pi, device=DEVICE, std=std, freeze_body=True,
    )


def global_eval_model(
    body_state: Dict,
    splits:     List[ClientSplitNLR],
    std:        C.Standardizer,
    cfg:        ExperimentConfig,
    seed:       int,
    use_pi:     bool,
) -> C.ElectrolyzerMLP:
    m = new_model(use_pi)
    m.set_body_state(body_state)
    m.reset_head(0.0)
    X_pool = np.vstack([cd.X_calib for cd in splits])
    y_pool = np.concatenate([std.transform_y(cd.y_calib) for cd in splits])
    I_pool = np.concatenate([cd.I_calib for cd in splits])
    C.train_local(
        model=m, X_norm=X_pool, y_norm=y_pool, I_phys=I_pool,
        n_epochs=cfg.eval_head_ep, lr_body=0.0, lr_head=cfg.lr_head,
        seed=seed + 7777, use_mono_loss=False, beta_mono=0.0,
        use_hard_guard=use_pi, device=DEVICE, std=std, freeze_body=True,
    )
    return m


def get_heads(
    models:     List[C.ElectrolyzerMLP],
    body_state: Optional[Dict] = None,
) -> Dict:
    bh = C.state_hash(body_state) if body_state is not None else None
    return {
        f"client_{i}": {
            "alpha_k":             float(m.alpha_k.detach().item()),
            "calibrated_for_body": bh,
        }
        for i, m in enumerate(models)
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_global(
    model:       C.ElectrolyzerMLP,
    std:         C.Standardizer,
    X_eval_norm: np.ndarray,
    y_eval_raw:  np.ndarray,
    I_eval_phys: np.ndarray,
    I_probe:     torch.Tensor,
    x_probe:     torch.Tensor,
) -> Dict[str, float]:
    gm = C.evaluate_model(
        model, X_eval_norm, y_eval_raw, I_eval_phys,
        std, I_probe, x_probe, DEVICE,
    )
    return {f"global_{k}": v for k, v in gm.items()}


def _client_eval_split(cd: ClientSplitNLR):
    """Return (X, y, I) using the mode set by load_data()."""
    if _EVAL_MODE == "final":
        return cd.X_test, cd.y_test, cd.I_test
    return cd.X_audit, cd.y_audit, cd.I_audit


def eval_per_client_personalised(
    models:  List[C.ElectrolyzerMLP],
    splits:  List[ClientSplitNLR],
    std:     C.Standardizer,
    I_probe: torch.Tensor,
    x_probe: torch.Tensor,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    per = [
        C.evaluate_model(
            models[i], *_client_eval_split(splits[i]),
            std, I_probe, x_probe, DEVICE,
        )
        for i in range(len(models))
    ]
    mac  = C.macro_metrics(per)
    flat = {f"client_{k}": v for k, v in mac.items()}
    flat.update(C.per_client_metrics_flat(per))
    return flat, per


def eval_shared_model_per_client(
    model:   C.ElectrolyzerMLP,
    splits:  List[ClientSplitNLR],
    std:     C.Standardizer,
    I_probe: torch.Tensor,
    x_probe: torch.Tensor,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    per = [
        C.evaluate_model(
            model, *_client_eval_split(cd),
            std, I_probe, x_probe, DEVICE,
        )
        for cd in splits
    ]
    mac  = C.macro_metrics(per)
    flat = {f"client_{k}": v for k, v in mac.items()}
    flat.update(C.per_client_metrics_flat(per))
    return flat, per


def eval_per_client_at_round(
    model:   C.ElectrolyzerMLP,
    splits:  List[ClientSplitNLR],
    std:     C.Standardizer,
    I_probe: torch.Tensor,
    x_probe: torch.Tensor,
    seed:    int,
    round_n: int,
) -> List[Dict]:
    rows = []
    for i, cd in enumerate(splits):
        X, y, I = _client_eval_split(cd)
        m = C.regression_metrics(model, X, y, I, std, DEVICE)
        p = C.physics_metrics(model, I_probe, x_probe, std, DEVICE)
        rows.append({
            "seed":     seed,
            "round":    round_n,
            "client":   i,
            "R2":       m["R2"],
            "RMSE":     m["RMSE"],
            "PVR":      p["mono_viol_rate"],
            "neg_rate": p["neg_rate"],
        })
    return rows


# ---------------------------------------------------------------------------
# Saving (identical structure to experiment_utils.py)
# ---------------------------------------------------------------------------

def _safe_for_json(obj):
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _safe_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_for_json(v) for v in obj]
    return obj


def save_run(
    out_dir:            str,
    method:             str,
    seed:               int,
    cfg:                ExperimentConfig,
    metrics:            Dict,
    round_log:          List[Dict],
    gate_log:           Optional[List[Dict]] = None,
    pf_log:             Optional[List[Dict]] = None,
    activity_log:       Optional[List[Dict]] = None,
    gate_scores_log:    Optional[List[Dict]] = None,
    per_client_log:     Optional[List[Dict]] = None,
    convergence_log:    Optional[List[Dict]] = None,
    final_body:         Optional[Dict]       = None,
    best_body:          Optional[Dict]       = None,
    personalized_heads: Optional[Dict]       = None,
    std:                Optional[C.Standardizer] = None,
    split_manifest:     Optional[Dict]       = None,
) -> None:
    safe_name = (method.replace(" ", "_").replace("/", "-")
                       .replace("(", "").replace(")", ""))
    folder = os.path.join(out_dir, safe_name, f"seed_{seed:03d}")
    os.makedirs(folder, exist_ok=True)

    cfg_d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    cfg_d["seeds"] = list(cfg_d["seeds"])
    C.save_json(cfg_d,                      os.path.join(folder, "config.json"))
    C.save_json(_safe_for_json(metrics),    os.path.join(folder, "final_metrics.json"))
    C.save_json({
        "python":    platform.python_version(),
        "torch":     torch.__version__,
        "numpy":     np.__version__,
        "device":    str(DEVICE),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }, os.path.join(folder, "environment.json"))

    logs = [
        (round_log,       "round_metrics.csv"),
        (gate_log,        "gate_log.csv"),
        (gate_scores_log, "gate_client_scores.csv"),
        (pf_log,          "forgetting_log.csv"),
        (activity_log,    "activity_log.csv"),
        (per_client_log,  "per_client_metrics.csv"),
        (convergence_log, "convergence_log.csv"),
    ]
    for log, fname in logs:
        if log:
            path = os.path.join(folder, fname)
            C.init_csv(path, list(log[0].keys()))
            for row in log:
                C.append_csv(path, row)

    state_hashes = {}
    if final_body is not None:
        torch.save(final_body, os.path.join(folder, "final_body.pt"))
        state_hashes["final_body"] = C.state_hash(final_body)
    if best_body is not None:
        torch.save(best_body, os.path.join(folder, "best_body.pt"))
        state_hashes["best_body"] = C.state_hash(best_body)
    if state_hashes:
        C.save_json(state_hashes, os.path.join(folder, "state_hashes.json"))

    if personalized_heads is not None:
        C.save_json(personalized_heads,
                    os.path.join(folder, "personalized_heads.json"))
    if std is not None:
        C.save_json(std.state_dict(), os.path.join(folder, "standardizer.json"))
    if split_manifest is not None:
        C.save_json(split_manifest, os.path.join(folder, "split_manifest.json"))


def aggregate_and_save_summary(
    agg:     Dict[str, List[Dict]],
    out_dir: str,
) -> None:
    summary: Dict = {}
    for name, results in agg.items():
        if not results:
            continue
        summary[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in results
                                       if k in r and r[k] is not None
                                       and np.isfinite(float(r[k]))])),
                "std":  float(np.std( [r[k] for r in results
                                       if k in r and r[k] is not None
                                       and np.isfinite(float(r[k]))])),
            }
            for k in results[0]
            if isinstance(results[0].get(k), (int, float))
            and results[0].get(k) is not None
        }
    C.save_json(_safe_for_json(summary), os.path.join(out_dir, "summary.json"))


def print_result_line(
    name:    str,
    metrics: Dict,
    cfg:     ExperimentConfig,
    gate:    bool = False,
) -> None:
    r2    = metrics.get("global_R2")
    pvr   = metrics.get("global_mono_viol_rate")
    r2_s  = "--" if r2  is None else f"{r2:.4f}"
    pvr_s = "--" if pvr is None else f"{pvr:.4f}"
    line  = f"  [{name}] R2={r2_s}  PVR={pvr_s}"
    if gate:
        acc   = metrics.get("n_accepted")
        line += f"  gate={acc}/{cfg.rounds}" if acc is not None else ""
    print(line)
