"""
common_wind_aei.py — Shared model, physics functions, and utilities for
the PI-SDFL wind AEI experiment.

Key differences from common_wind.py (ESWA):
  - TAU_PVR = 0.05  (restored to match synthetic PEM; was incorrectly 0.10)
  - Physics probe loaded from pre-built CSV (same design as NLR)
  - Standardizer loaded from JSON (Client 0 / WT3 train only)
  - Terminology: "below-rated wind-speed-power monotonicity prior"
    NOT "Betz's law enforcement"
  - No reference to "coordinator" — Client 0 is the initial/warm-start client

Physics law (what is actually enforced):
  dP/dV >= 0 in the below-rated operating region (1%–95% rated power).
  The hard guard guarantees P >= 0 (non-negativity).
  Monotonicity is encouraged via L_mono regularisation only —
  NOT architecturally guaranteed and NOT a full cubic Betz power law.

Do not import common_wind.py from src_wind_aei runners. Import this file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES: List[str] = [
    "wind_speed",    # [0] physics feature — dP/dV >= 0
    "wind_dir",      # [1] direction — frozen in probe
    "air_density",   # [2] kg/m³ — context feature in probe
    "turbulence",    # [3] dimensionless — context feature in probe
    "wind_shear",    # [4] exponent — frozen in probe
]
TARGET:     str = "power_pct"
N_FEATURES: int = len(FEATURES)   # 5

HIDDEN_DIM: int = 64
NUM_HIDDEN: int = 3
OUTPUT_DIM: int = 1

EPSILON_MONO: float = 1e-3
TAU_PVR:      float = 0.05   # restored — same as synthetic PEM
TAU_MAG:      float = 0.10
EPSILON_R2:   float = 0.02
CWC_LAMBDA:   float = 0.1
BATCH_SIZE:   int   = 64


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Standardizer (loaded from JSON — never re-fitted at run time)
# ---------------------------------------------------------------------------

class Standardizer:
    """
    Fitted on Client 0 (WT3) training data by run_wind_preprocessing.py.
    Loaded from standardizer_client0_train.json; never re-fitted at run time.
    """

    def __init__(self) -> None:
        self.X_mean:  Optional[np.ndarray] = None
        self.X_std:   Optional[np.ndarray] = None
        self.y_mean:  Optional[float]      = None
        self.y_std:   Optional[float]      = None
        self._fitted: bool                 = False

    @classmethod
    def from_json(cls, json_path: str) -> "Standardizer":
        with open(json_path) as f:
            d = json.load(f)
        s = cls()
        s.X_mean  = np.array([d["X_mean"][f] for f in FEATURES], dtype=np.float32)
        s.X_std   = np.array([d["X_std"][f]  for f in FEATURES], dtype=np.float32)
        s.y_mean  = float(d["y_mean"])
        s.y_std   = float(d["y_std"])
        s._fitted = True
        return s

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Standardizer not loaded. Call Standardizer.from_json().")

    def transform_X(self, X_raw: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((X_raw - self.X_mean) / self.X_std).astype(np.float32)

    def transform_y(self, y_raw: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((y_raw.ravel() - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (y_norm.ravel() * self.y_std + self.y_mean).astype(np.float32)

    @property
    def PHYS_mean(self) -> float:
        self._check_fitted()
        return float(self.X_mean[0])   # wind_speed mean

    @property
    def PHYS_std(self) -> float:
        self._check_fitted()
        return float(self.X_std[0])    # wind_speed std

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "X_mean": self.X_mean.tolist(),
            "X_std":  self.X_std.tolist(),
            "y_mean": self.y_mean,
            "y_std":  self.y_std,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Standardizer":
        s = cls()
        s.X_mean  = np.array(d["X_mean"], dtype=np.float32)
        s.X_std   = np.array(d["X_std"],  dtype=np.float32)
        s.y_mean  = float(d["y_mean"])
        s.y_std   = float(d["y_std"])
        s._fitted = True
        return s


# ---------------------------------------------------------------------------
# Model — 5-feature wind body, identical head and forward API to NLR
# ---------------------------------------------------------------------------

class WindTurbineMLP(nn.Module):
    """
    Body: Linear(5->64)->ReLU x3 -> Linear(64->1)  — shared across ring.
    Head: scalar alpha_k per client — stays local.

    Physics law: below-rated wind-speed-power monotonicity prior (dP/dV >= 0).
    The hard guard guarantees P >= 0 (non-negativity via Softplus × clamp(V,0)).
    Monotonicity is encouraged via L_mono regularisation only.

    Hard guard (use_hard_guard=True):
        P_phys = P_ref × Softplus(alpha_k + body_out) × (V_plus / V_ref)
        output = (P_phys - P_ref) / y_std

    Generic MLP (use_hard_guard=False):
        output = alpha_k + body_out

    forward() parameter names kept as I_ref/H_ref for API compatibility
    with common_nlr.py:
        I_ref = V_ref = wind_speed mean (Standardizer.PHYS_mean)
        H_ref = P_ref = power_pct mean  (Standardizer.y_mean)
    """

    def __init__(self, use_hard_guard: bool = True) -> None:
        super().__init__()
        self.use_hard_guard = use_hard_guard
        layers: List[nn.Module] = []
        in_dim = N_FEATURES
        for _ in range(NUM_HIDDEN):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.ReLU())
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(in_dim, OUTPUT_DIM))
        self.body    = nn.Sequential(*layers)
        self.alpha_k = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        I_phys: torch.Tensor,   # raw wind_speed in m/s
        x_norm: torch.Tensor,   # normalized [V,D,rho,I,S] (batch, 5)
        I_ref:  float = 1.0,    # V_ref = Standardizer.PHYS_mean
        H_ref:  float = 1.0,    # P_ref = Standardizer.y_mean
        y_std:  float = 1.0,
    ) -> torch.Tensor:
        if I_phys.dim() == 1:
            I_phys = I_phys.unsqueeze(1)
        body_out = self.body(x_norm)
        if self.use_hard_guard:
            V_plus  = torch.clamp(I_phys, min=0.0)
            P_phys  = H_ref * F.softplus(self.alpha_k + body_out) * (V_plus / I_ref)
            return ((P_phys - H_ref) / y_std).squeeze(1)
        else:
            return (self.alpha_k + body_out).squeeze(1)

    def get_body_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone()
                for k, v in self.body.state_dict().items()}

    def set_body_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.body.load_state_dict(
            {k: v.clone() for k, v in state.items()}, strict=True)

    def get_head_state(self) -> Dict[str, float]:
        return {"alpha_k": float(self.alpha_k.item())}

    def set_head_state(self, state: Dict[str, float]) -> None:
        with torch.no_grad():
            self.alpha_k.copy_(torch.tensor(state["alpha_k"], dtype=torch.float32))

    def reset_head(self, value: float = 0.0) -> None:
        with torch.no_grad():
            self.alpha_k.fill_(value)

    def count_body_params(self) -> int:
        return sum(p.numel() for p in self.body.parameters())


# ---------------------------------------------------------------------------
# State dict utilities
# ---------------------------------------------------------------------------

def clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_hash(state: Dict[str, torch.Tensor]) -> str:
    flat = torch.cat([v.detach().cpu().flatten() for v in state.values()])
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:8]


def weighted_average_states(
    states:        List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    total   = float(sum(sample_counts))
    weights = [c / total for c in sample_counts]
    avg: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        avg[key] = sum(w * s[key].float() for w, s in zip(weights, states)
                       ).to(states[0][key].dtype)
    return avg


def equal_average_states(
    states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    n = len(states)
    avg: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        avg[key] = (sum(s[key].float() for s in states) / n
                    ).to(states[0][key].dtype)
    return avg


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_split_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-normalized wind split CSV.
    Returns (X_norm [N, 5], y_norm [N]) — float32.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    return (df[FEATURES].values.astype(np.float32),
            df[TARGET].values.astype(np.float32))


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Physics probe — loaded from pre-built CSV
# ---------------------------------------------------------------------------

def load_probe(
    norm_probe_path: str,
    raw_probe_path:  str,
    device:          torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the pre-built wind physics probe.

    Returns:
        I_probe_phys : [250] float32 — physical wind_speed values in m/s
        x_probe_norm : [250, 5] float32 — normalized feature rows
            col 0 = normalized wind_speed (varies across sweep)
            col 1 = wind_dir (frozen at training median)
            col 2 = air_density (context quantile)
            col 3 = turbulence  (context quantile)
            col 4 = wind_shear  (frozen at training median)
    """
    norm_df = pd.read_csv(norm_probe_path)
    raw_df  = pd.read_csv(raw_probe_path)
    x_probe_norm = norm_df[FEATURES].values.astype(np.float32)
    I_probe_phys = raw_df["wind_speed"].values.astype(np.float32)
    return (
        torch.tensor(I_probe_phys, dtype=torch.float32, device=device),
        torch.tensor(x_probe_norm, dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_dataloader(
    X_norm:  np.ndarray,
    y_norm:  np.ndarray,
    I_phys:  np.ndarray,
    seed:    int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(I_phys, dtype=torch.float32),
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(y_norm, dtype=torch.float32),
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=make_dataloader_generator(seed), drop_last=False,
    )


# ---------------------------------------------------------------------------
# Monotonicity loss
# ---------------------------------------------------------------------------

def monotonicity_loss_epoch(
    model:        WindTurbineMLP,
    I_probe_phys: torch.Tensor,
    x_probe_norm: torch.Tensor,
    std:          Standardizer,
    epsilon:      float = EPSILON_MONO,
) -> torch.Tensor:
    """
    L_mono = mean( ReLU(-dP/dV - epsilon) )

    wind_speed (col 0) varied via autograd.
    Cols 1-4 (wind_dir, air_density, turbulence, wind_shear) frozen.
    """
    V_ref_val = float(std.PHYS_mean)
    V_std_val = float(std.PHYS_std)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    V_var      = I_probe_phys.detach().requires_grad_(True)
    V_norm_var = (V_var - V_ref_val) / V_std_val

    X_p = torch.cat([
        V_norm_var.unsqueeze(1),
        x_probe_norm[:, 1:].detach(),
    ], dim=1)

    model.train()
    P_std = model(V_var, X_p, I_ref=V_ref_val, H_ref=H_ref_val, y_std=y_std_val)
    grads = torch.autograd.grad(P_std.sum(), V_var, create_graph=True)[0]
    return F.relu(-grads - epsilon).mean()


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def _anchor_loss(
    model:         WindTurbineMLP,
    anchor_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    diffs      = []
    body_state = dict(model.body.named_parameters())
    for k, ref in anchor_params.items():
        short_k = k.replace("body.", "") if k.startswith("body.") else k
        if short_k in body_state:
            diffs.append((body_state[short_k] - ref).pow(2).mean())
    if not diffs:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(diffs).mean()


def train_local(
    model:          WindTurbineMLP,
    X_norm:         np.ndarray,
    y_norm:         np.ndarray,
    I_phys:         np.ndarray,
    n_epochs:       int,
    lr_body:        float,
    lr_head:        float,
    seed:           int,
    use_mono_loss:  bool,
    beta_mono:      float,
    use_hard_guard: bool,
    device:         torch.device,
    std:            Standardizer,
    I_probe_phys:   Optional[torch.Tensor] = None,
    x_probe_norm:   Optional[torch.Tensor] = None,
    anchor_body:    Optional[Dict[str, torch.Tensor]] = None,
    mu_anchor:      float = 0.0,
    freeze_body:    bool  = False,
) -> Dict[str, float]:
    if use_mono_loss and (I_probe_phys is None or x_probe_norm is None):
        raise ValueError("use_mono_loss=True requires I_probe_phys and x_probe_norm.")
    if std is None:
        raise ValueError("train_local requires a fitted Standardizer.")
    if use_hard_guard != model.use_hard_guard:
        raise ValueError(
            f"Architecture mismatch: use_hard_guard={use_hard_guard} "
            f"vs model.use_hard_guard={model.use_hard_guard}.")

    model.to(device).train()
    V_ref_val = float(std.PHYS_mean)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    loader    = make_dataloader(X_norm, y_norm, I_phys, seed=seed, shuffle=True)
    body_lr   = 0.0 if freeze_body else lr_body
    optimizer = torch.optim.Adam([
        {"params": model.body.parameters(), "lr": body_lr},
        {"params": [model.alpha_k],          "lr": lr_head},
    ])

    anchor_params = None
    if anchor_body is not None and mu_anchor > 0.0:
        anchor_params = {k: v.to(device).detach() for k, v in anchor_body.items()}

    l_total_ep = l_data_ep = l_mono_ep = l_anchor_ep = 0.0

    for _ in range(n_epochs):
        l_data_batches = []
        l_mono_val = l_anchor_val = 0.0
        batches   = list(loader)
        n_batches = len(batches)

        for b_idx, (V_b, X_b, y_b) in enumerate(batches):
            V_b, X_b, y_b = V_b.to(device), X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            y_pred = model(V_b, X_b, I_ref=V_ref_val, H_ref=H_ref_val, y_std=y_std_val)
            l_data  = F.mse_loss(y_pred, y_b)
            is_last = (b_idx == n_batches - 1)

            if is_last:
                reg_loss = torch.tensor(0.0, device=device)
                if use_mono_loss and beta_mono > 0.0:
                    l_mono_t   = monotonicity_loss_epoch(
                        model, I_probe_phys, x_probe_norm, std=std)
                    l_mono_val = l_mono_t.item()
                    reg_loss   = reg_loss + beta_mono * l_mono_t
                if anchor_params is not None:
                    l_anc_t      = _anchor_loss(model, anchor_params)
                    l_anchor_val = l_anc_t.item()
                    reg_loss     = reg_loss + mu_anchor * l_anc_t
                (l_data + reg_loss).backward()
            else:
                l_data.backward()

            optimizer.step()
            l_data_batches.append(l_data.item())

        l_data_ep   = float(np.mean(l_data_batches))
        l_mono_ep   = l_mono_val
        l_anchor_ep = l_anchor_val
        l_total_ep  = l_data_ep + beta_mono * l_mono_ep + mu_anchor * l_anchor_ep

    return {"l_total": l_total_ep, "l_data": l_data_ep,
            "l_mono": l_mono_ep,  "l_anchor": l_anchor_ep}


# ---------------------------------------------------------------------------
# Physics metrics
# ---------------------------------------------------------------------------

def physics_metrics(
    model:        WindTurbineMLP,
    I_probe_phys: torch.Tensor,
    x_probe_norm: torch.Tensor,
    std:          Standardizer,
    device:       torch.device,
    epsilon:      float = EPSILON_MONO,
) -> Dict[str, float]:
    model.eval()
    V_ref_val = float(std.PHYS_mean)
    V_std_val = float(std.PHYS_std)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    V_var      = I_probe_phys.detach().to(device).requires_grad_(True)
    V_norm_var = (V_var - V_ref_val) / V_std_val

    X_p = torch.cat([
        V_norm_var.unsqueeze(1),
        x_probe_norm[:, 1:].detach().to(device),
    ], dim=1)

    with torch.enable_grad():
        P_std = model(V_var, X_p, I_ref=V_ref_val, H_ref=H_ref_val, y_std=y_std_val)
        grads = torch.autograd.grad(P_std.sum(), V_var, create_graph=False)[0].detach()

    viol_mask = grads < -epsilon
    n_viol    = int(viol_mask.sum().item())
    M         = grads.numel()

    mono_viol_rate = n_viol / M
    if n_viol > 0:
        viol_grads              = grads[viol_mask]
        mono_viol_magnitude     = float(viol_grads.abs().mean().item())
        mono_viol_magnitude_max = float(viol_grads.abs().max().item())
    else:
        mono_viol_magnitude     = 0.0
        mono_viol_magnitude_max = 0.0

    P_phys   = P_std.detach() * y_std_val + H_ref_val
    neg_rate = float((P_phys < 0.0).sum().item()) / M

    return {
        "mono_viol_rate":          mono_viol_rate,
        "mono_viol_magnitude":     mono_viol_magnitude,
        "mono_viol_magnitude_max": mono_viol_magnitude_max,
        "neg_rate":                neg_rate,
    }


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    model:  WindTurbineMLP,
    X_norm: np.ndarray,
    y_raw:  np.ndarray,
    I_phys: np.ndarray,
    std:    Standardizer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        Vt = torch.tensor(I_phys, dtype=torch.float32).to(device)
        Xt = torch.tensor(X_norm, dtype=torch.float32).to(device)
        y_pred_norm = model(
            Vt, Xt, I_ref=std.PHYS_mean, H_ref=std.y_mean, y_std=std.y_std
        ).cpu().numpy()
    y_pred_phys = std.inverse_y(y_pred_norm)
    y_true_phys = y_raw.ravel().astype(np.float32)
    return {
        "R2":   float(r2_score(y_true_phys, y_pred_phys)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_phys, y_pred_phys))),
        "MAE":  float(mean_absolute_error(y_true_phys, y_pred_phys)),
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model:        WindTurbineMLP,
    X_norm:       np.ndarray,
    y_raw:        np.ndarray,
    I_phys:       np.ndarray,
    std:          Standardizer,
    I_probe_phys: torch.Tensor,
    x_probe_norm: torch.Tensor,
    device:       torch.device,
) -> Dict[str, float]:
    reg  = regression_metrics(model, X_norm, y_raw, I_phys, std, device)
    phys = physics_metrics(model, I_probe_phys, x_probe_norm, std, device)
    rate = phys["mono_viol_rate"]
    mag  = min(phys["mono_viol_magnitude"], 1.0)
    pc_r2 = reg["R2"] * (1.0 - rate) * (1.0 - mag)
    return {
        "R2":                  reg["R2"],
        "RMSE":                reg["RMSE"],
        "MAE":                 reg["MAE"],
        "mono_viol_rate":      phys["mono_viol_rate"],
        "mono_viol_magnitude": phys["mono_viol_magnitude"],
        "mono_viol_mag_max":   phys["mono_viol_magnitude_max"],
        "neg_rate":            phys["neg_rate"],
        "PC_R2":               round(pc_r2, 6),
    }


def macro_metrics(per_client: List[Dict[str, float]]) -> Dict[str, float]:
    lower_is_worse = {"R2", "PC_R2"}
    keys = ["R2", "RMSE", "MAE",
            "mono_viol_rate", "mono_viol_magnitude", "mono_viol_mag_max",
            "neg_rate", "PC_R2"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in per_client]
        out[f"{k}_macro"] = float(np.mean(vals))
        out[f"{k}_worst"] = float(np.min(vals) if k in lower_is_worse
                                  else np.max(vals))
    return out


def per_client_metrics_flat(
    per_client: List[Dict[str, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, m in enumerate(per_client):
        out[f"c{i}_R2"]    = m["R2"]
        out[f"c{i}_RMSE"]  = m["RMSE"]
        out[f"c{i}_PVR"]   = m["mono_viol_rate"]
        out[f"c{i}_neg"]   = m["neg_rate"]
        out[f"c{i}_PC_R2"] = m["PC_R2"]
    return out


# ---------------------------------------------------------------------------
# Gate scoring
# ---------------------------------------------------------------------------

def score_candidate(
    body_state:     Dict[str, torch.Tensor],
    head_init_val:  float,
    X_calib_norm:   np.ndarray,
    y_calib_raw:    np.ndarray,
    I_calib_phys:   np.ndarray,
    X_audit_norm:   np.ndarray,
    y_audit_raw:    np.ndarray,
    I_audit_phys:   np.ndarray,
    I_probe_phys:   torch.Tensor,
    x_probe_norm:   torch.Tensor,
    std:            Standardizer,
    device:         torch.device,
    seed:           int,
    calib_epochs:   int,
    lr_head:        float,
    use_hard_guard: bool,
) -> Dict[str, float]:
    tmp = WindTurbineMLP(use_hard_guard=use_hard_guard).to(device)
    tmp.set_body_state(body_state)
    tmp.reset_head(value=head_init_val)
    y_calib_norm = std.transform_y(y_calib_raw)
    train_local(
        model=tmp, X_norm=X_calib_norm, y_norm=y_calib_norm,
        I_phys=I_calib_phys, n_epochs=calib_epochs,
        lr_body=0.0, lr_head=lr_head, seed=seed,
        use_mono_loss=False, beta_mono=0.0,
        use_hard_guard=use_hard_guard, device=device, std=std, freeze_body=True,
    )
    return evaluate_model(
        model=tmp, X_norm=X_audit_norm, y_raw=y_audit_raw,
        I_phys=I_audit_phys, std=std,
        I_probe_phys=I_probe_phys, x_probe_norm=x_probe_norm, device=device,
    )


# ---------------------------------------------------------------------------
# Trust gate
# ---------------------------------------------------------------------------

def trust_gate_decision(
    candidate_metrics: Dict[str, float],
    r2_best:           float,
    tau_pvr:           float = TAU_PVR,
    tau_mag:           float = TAU_MAG,
    epsilon_r2:        float = EPSILON_R2,
) -> Dict:
    if "R2" in candidate_metrics:
        r2  = float(candidate_metrics["R2"])
        pvr = float(candidate_metrics["mono_viol_rate"])
        mag = float(candidate_metrics["mono_viol_magnitude"])
    elif "R2_macro" in candidate_metrics:
        r2  = float(candidate_metrics["R2_macro"])
        pvr = float(candidate_metrics["mono_viol_rate_worst"])
        mag = float(candidate_metrics["mono_viol_magnitude_worst"])
    else:
        raise KeyError(
            f"candidate_metrics must contain 'R2' or 'R2_macro'. "
            f"Got: {sorted(candidate_metrics.keys())}")

    physics_ok   = pvr <= tau_pvr
    magnitude_ok = mag <= tau_mag
    accuracy_ok  = r2  >= r2_best - epsilon_r2
    accept       = physics_ok and magnitude_ok and accuracy_ok

    reasons = []
    if not physics_ok:   reasons.append(f"PVR={pvr:.4f}>{tau_pvr}")
    if not magnitude_ok: reasons.append(f"Mag={mag:.4f}>{tau_mag}")
    if not accuracy_ok:  reasons.append(f"R2={r2:.4f}<{r2_best - epsilon_r2:.4f}")
    reason_str = "ACCEPT" if accept else f"REJECT ({', '.join(reasons)})"

    return {
        "accept":       accept,
        "physics_ok":   physics_ok,
        "magnitude_ok": magnitude_ok,
        "accuracy_ok":  accuracy_ok,
        "r2_cand":      round(r2,  6),
        "pvr_cand":     round(pvr, 6),
        "mag_cand":     round(mag, 6),
        "reason":       reason_str,
    }


# ---------------------------------------------------------------------------
# Physics forgetting
# ---------------------------------------------------------------------------

def body_forgetting(
    body_now:       Dict[str, torch.Tensor],
    body_ref:       Dict[str, torch.Tensor],
    head_ref_val:   float,
    X_audit_norm:   np.ndarray,
    y_audit_raw:    np.ndarray,
    I_audit_phys:   np.ndarray,
    I_probe_phys:   torch.Tensor,
    x_probe_norm:   torch.Tensor,
    std:            Standardizer,
    device:         torch.device,
    use_hard_guard: bool,
) -> Dict[str, float]:
    def _pvr_and_r2(body_state):
        tmp = WindTurbineMLP(use_hard_guard=use_hard_guard).to(device)
        tmp.set_body_state(body_state)
        tmp.reset_head(value=head_ref_val)
        phys = physics_metrics(tmp, I_probe_phys, x_probe_norm, std, device)
        reg  = regression_metrics(tmp, X_audit_norm, y_audit_raw,
                                  I_audit_phys, std, device)
        return phys["mono_viol_rate"], reg["R2"]

    pvr_ref, r2_ref = _pvr_and_r2(body_ref)
    pvr_now, r2_now = _pvr_and_r2(body_now)
    pf_signed       = pvr_now - pvr_ref
    return {
        "pvr_ref":     pvr_ref,
        "pvr_now":     pvr_now,
        "pf_signed":   pf_signed,
        "pf_positive": max(0.0, pf_signed),
        "r2_ref":      r2_ref,
        "r2_now":      r2_now,
    }


def aggregate_forgetting(pf_records: List[Dict[str, float]]) -> Dict[str, float]:
    if not pf_records:
        return {"apf_signed": 0.0, "apf_positive": 0.0,
                "pf_peak": 0.0, "n_forgetting": 0, "n_total_steps": 0}
    signed   = [r["pf_signed"]   for r in pf_records]
    positive = [r["pf_positive"] for r in pf_records]
    return {
        "apf_signed":    float(np.mean(signed)),
        "apf_positive":  float(np.mean(positive)),
        "pf_peak":       float(np.max(positive)),
        "n_forgetting":  int(sum(1 for s in signed if s > 0)),
        "n_total_steps": len(pf_records),
    }


# ---------------------------------------------------------------------------
# CWC utilities
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(
    model:     WindTurbineMLP,
    X_norm:    np.ndarray,
    y_norm:    np.ndarray,
    I_phys:    np.ndarray,
    std:       Standardizer,
    device:    torch.device,
    n_samples: int = 200,
    seed:      int = 0,
) -> Dict[str, torch.Tensor]:
    model.eval()
    rng    = np.random.default_rng(seed)
    fisher = {name: torch.zeros_like(param)
              for name, param in model.body.named_parameters()}
    N   = len(X_norm)
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    for i in idx:
        model.zero_grad()
        Vt     = torch.tensor(I_phys[i:i+1], dtype=torch.float32).to(device)
        Xt     = torch.tensor(X_norm[i:i+1], dtype=torch.float32).to(device)
        yt     = torch.tensor(y_norm[i:i+1], dtype=torch.float32).to(device)
        y_pred = model(Vt, Xt, I_ref=std.PHYS_mean, H_ref=std.y_mean, y_std=std.y_std)
        F.mse_loss(y_pred, yt).backward()
        for name, param in model.body.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
    for name in fisher:
        fisher[name] /= min(n_samples, N)
    return fisher


def cwc_penalty(
    model:          WindTurbineMLP,
    ref_body_state: Dict[str, torch.Tensor],
    fisher:         Dict[str, torch.Tensor],
    device:         torch.device,
) -> torch.Tensor:
    penalty     = torch.tensor(0.0, device=device, requires_grad=False)
    body_params = dict(model.body.named_parameters())
    for name, ref_val in ref_body_state.items():
        short = name.replace("body.", "")
        if short in body_params and short in fisher:
            diff    = body_params[short] - ref_val.to(device)
            penalty = penalty + (fisher[short].to(device) * diff.pow(2)).sum()
    return penalty


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_json(data: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def init_csv(filepath: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv(filepath: str, row: dict) -> None:
    with open(filepath, "r", newline="") as f:
        fieldnames = next(csv.reader(f))
    if set(row.keys()) != set(fieldnames):
        raise ValueError(
            f"append_csv: row keys {sorted(row.keys())} do not match "
            f"header {sorted(fieldnames)}.")
    with open(filepath, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def separator(title: str = "", width: int = 70) -> None:
    line = "=" * width
    print(f"\n{line}\n  {title}\n{line}" if title else line)
