import os
import csv
import json
import hashlib
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES:   List[str] = ["current", "temperature", "pressure"]
TARGET:     str       = "h2_flow_rate"
N_FEATURES: int       = len(FEATURES)

HIDDEN_DIM: int = 64
NUM_HIDDEN: int = 3
OUTPUT_DIM: int = 1

PHYS_BOUNDS = {
    "current":     (14.0, 160.0),
    "temperature": (38.0,  80.0),
    "pressure":    ( 0.8,   7.0),
}

N_PROBE_I:    int   = 50
N_PROBE_T:    int   = 5
N_PROBE_P:    int   = 5
EPSILON_MONO: float = 1e-3

TAU_PVR:    float = 0.05
TAU_MAG:    float = 0.10
EPSILON_R2: float = 0.02
CWC_LAMBDA: float = 0.1
BATCH_SIZE: int   = 64


# ── Seeding ───────────────────────────────────────────────────────────────────

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


# ── Standardizer ──────────────────────────────────────────────────────────────

class Standardizer:
    """
    Fitted on Client 1 training data only.
    All clients and global evaluation use the same fitted instance.
    """

    def __init__(self) -> None:
        self.X_mean: Optional[np.ndarray] = None
        self.X_std:  Optional[np.ndarray] = None
        self.y_mean: Optional[float]      = None
        self.y_std:  Optional[float]      = None
        self._fitted: bool                = False

    def fit(self, X_raw: np.ndarray, y_raw: np.ndarray) -> "Standardizer":
        y_flat = y_raw.ravel()
        self.X_mean = X_raw.mean(axis=0).astype(np.float32)
        self.X_std  = X_raw.std(axis=0).astype(np.float32)
        self.y_mean = float(y_flat.mean())
        self.y_std  = float(y_flat.std())
        if np.any(self.X_std < 1e-8):
            raise ValueError(f"Near-zero std in features: {self.X_std}")
        if self.y_std < 1e-8:
            raise ValueError("Near-zero std in target.")
        if self.X_mean[0] <= 0:
            raise ValueError(f"I_mean={self.X_mean[0]:.4f} must be > 0.")
        if self.y_mean <= 0:
            raise ValueError(f"y_mean={self.y_mean:.4f} must be > 0.")
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Standardizer not fitted. Call .fit() first.")

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
    def I_mean(self) -> float:
        self._check_fitted(); return float(self.X_mean[0])

    @property
    def I_std(self) -> float:
        self._check_fitted(); return float(self.X_std[0])

    @property
    def T_mean(self) -> float:
        self._check_fitted(); return float(self.X_mean[1])

    @property
    def T_std(self) -> float:
        self._check_fitted(); return float(self.X_std[1])

    @property
    def P_mean(self) -> float:
        self._check_fitted(); return float(self.X_mean[2])

    @property
    def P_std(self) -> float:
        self._check_fitted(); return float(self.X_std[2])

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


# ── Model ─────────────────────────────────────────────────────────────────────

class ElectrolyzerMLP(nn.Module):
    """
    Shared model class for all experiment variants.

    Body: Linear(3->64)->ReLU x3 -> Linear(64->1)  — travels the ring.
    Head: scalar alpha_k per client — stays local, never transmitted.

    use_hard_guard=True  (PI architecture):
        H2_pred = H_ref * Softplus(alpha_k + body_out) * (I_plus / I_ref)
        Guarantees non-negative predictions. Monotonicity is soft (L_mono).

    use_hard_guard=False (generic MLP):
        H2_pred = alpha_k + body_out   (no non-negativity guarantee)
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
        I_phys: torch.Tensor,
        x_norm: torch.Tensor,
        I_ref:  float = 1.0,
        H_ref:  float = 1.0,
        y_std:  float = 1.0,
    ) -> torch.Tensor:
        if I_phys.dim() == 1:
            I_phys = I_phys.unsqueeze(1)
        body_out = self.body(x_norm)
        if self.use_hard_guard:
            I_plus  = torch.clamp(I_phys, min=0.0)
            H2_phys = H_ref * F.softplus(self.alpha_k + body_out) * (I_plus / I_ref)
            return ((H2_phys - H_ref) / y_std).squeeze(1)
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


# ── State dict utilities ──────────────────────────────────────────────────────

def clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_hash(state: Dict[str, torch.Tensor]) -> str:
    flat = torch.cat([v.detach().cpu().flatten() for v in state.values()])
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:8]


def weighted_average_states(
    states: List[Dict[str, torch.Tensor]],
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
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


# ── DataLoader ────────────────────────────────────────────────────────────────

def make_dataloader(
    X_norm: np.ndarray,
    y_norm: np.ndarray,
    I_phys: np.ndarray,
    seed:   int,
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


# ── Physics probe grid ────────────────────────────────────────────────────────

def build_probe_grid(
    std:          Standardizer,
    device:       torch.device,
    client_X_raw: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    I_lo, I_hi = PHYS_BOUNDS["current"]
    if client_X_raw is not None:
        T_lo = float(client_X_raw[:, 1].min())
        T_hi = float(client_X_raw[:, 1].max())
        P_lo = float(client_X_raw[:, 2].min())
        P_hi = float(client_X_raw[:, 2].max())
        if T_hi - T_lo < 1.0:
            T_lo, T_hi = PHYS_BOUNDS["temperature"]
        if P_hi - P_lo < 0.1:
            P_lo, P_hi = PHYS_BOUNDS["pressure"]
    else:
        T_lo, T_hi = PHYS_BOUNDS["temperature"]
        P_lo, P_hi = PHYS_BOUNDS["pressure"]

    I_grid = np.linspace(I_lo, I_hi, N_PROBE_I)
    T_grid = np.linspace(T_lo, T_hi, N_PROBE_T)
    P_grid = np.linspace(P_lo, P_hi, N_PROBE_P)
    II, TT, PP = np.meshgrid(I_grid, T_grid, P_grid, indexing="ij")
    I_flat = II.ravel().astype(np.float32)
    T_flat = TT.ravel().astype(np.float32)
    P_flat = PP.ravel().astype(np.float32)

    X_probe_phys = np.stack([I_flat, T_flat, P_flat], axis=1)
    X_probe_norm = std.transform_X(X_probe_phys)

    return (torch.tensor(I_flat, dtype=torch.float32, device=device),
            torch.tensor(X_probe_norm, dtype=torch.float32, device=device))


# ── Monotonicity loss ─────────────────────────────────────────────────────────

def monotonicity_loss_epoch(
    model:        ElectrolyzerMLP,
    I_probe_phys: torch.Tensor,
    x_probe_norm: torch.Tensor,
    std:          Standardizer,
    epsilon:      float = EPSILON_MONO,
) -> torch.Tensor:
    I_ref_val = float(std.I_mean)
    I_std_val = float(std.I_std)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    I_var      = I_probe_phys.detach().requires_grad_(True)
    I_norm_var = (I_var - I_ref_val) / I_std_val
    X_p = torch.stack([I_norm_var,
                        x_probe_norm[:, 1].detach(),
                        x_probe_norm[:, 2].detach()], dim=1)
    model.train()
    H2    = model(I_var, X_p, I_ref=I_ref_val, H_ref=H_ref_val, y_std=y_std_val)
    grads = torch.autograd.grad(H2.sum(), I_var, create_graph=True)[0]
    return F.relu(-grads - epsilon).mean()


# ── Local training ────────────────────────────────────────────────────────────

def train_local(
    model:          ElectrolyzerMLP,
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
    std:            Standardizer = None,
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
    I_ref_val = float(std.I_mean)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    loader  = make_dataloader(X_norm, y_norm, I_phys, seed=seed, shuffle=True)
    body_lr = 0.0 if freeze_body else lr_body
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

        for b_idx, (I_b, X_b, y_b) in enumerate(batches):
            I_b, X_b, y_b = I_b.to(device), X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            y_pred = model(I_b, X_b, I_ref=I_ref_val, H_ref=H_ref_val, y_std=y_std_val)
            l_data = F.mse_loss(y_pred, y_b)
            is_last = (b_idx == n_batches - 1)

            if is_last:
                reg_loss = torch.tensor(0.0, device=device)
                if use_mono_loss and beta_mono > 0.0:
                    l_mono_t   = monotonicity_loss_epoch(model, I_probe_phys, x_probe_norm, std=std)
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


def _anchor_loss(
    model:         ElectrolyzerMLP,
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


# ── Physics metrics ───────────────────────────────────────────────────────────

def physics_metrics(
    model:        ElectrolyzerMLP,
    I_probe_phys: torch.Tensor,
    x_probe_norm: torch.Tensor,
    std:          Standardizer,
    device:       torch.device,
    epsilon:      float = EPSILON_MONO,
) -> Dict[str, float]:
    model.eval()
    I_ref_val = float(std.I_mean)
    I_std_val = float(std.I_std)
    H_ref_val = float(std.y_mean)
    y_std_val = float(std.y_std)

    I_var      = I_probe_phys.detach().to(device).requires_grad_(True)
    I_norm_var = (I_var - I_ref_val) / I_std_val
    X_p = torch.stack([I_norm_var,
                        x_probe_norm[:, 1].detach().to(device),
                        x_probe_norm[:, 2].detach().to(device)], dim=1)

    with torch.enable_grad():
        H2_std = model(I_var, X_p, I_ref=I_ref_val, H_ref=H_ref_val, y_std=y_std_val)
        grads  = torch.autograd.grad(H2_std.sum(), I_var, create_graph=False)[0].detach()

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

    H2_phys  = H2_std.detach() * y_std_val + H_ref_val
    neg_rate = float((H2_phys < 0.0).sum().item()) / M

    return {
        "mono_viol_rate":          mono_viol_rate,
        "mono_viol_magnitude":     mono_viol_magnitude,
        "mono_viol_magnitude_max": mono_viol_magnitude_max,
        "neg_rate":                neg_rate,
    }


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(
    model:  ElectrolyzerMLP,
    X_norm: np.ndarray,
    y_raw:  np.ndarray,
    I_phys: np.ndarray,
    std:    Standardizer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        It = torch.tensor(I_phys, dtype=torch.float32).to(device)
        Xt = torch.tensor(X_norm, dtype=torch.float32).to(device)
        y_pred_norm = model(It, Xt, I_ref=std.I_mean,
                            H_ref=std.y_mean, y_std=std.y_std).cpu().numpy()
    y_pred_phys = std.inverse_y(y_pred_norm)
    y_true_phys = y_raw.ravel().astype(np.float32)
    return {
        "R2":   float(r2_score(y_true_phys, y_pred_phys)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_phys, y_pred_phys))),
        "MAE":  float(mean_absolute_error(y_true_phys, y_pred_phys)),
    }


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    model:        ElectrolyzerMLP,
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
        "R2":                    reg["R2"],
        "RMSE":                  reg["RMSE"],
        "MAE":                   reg["MAE"],
        "mono_viol_rate":        phys["mono_viol_rate"],
        "mono_viol_magnitude":   phys["mono_viol_magnitude"],
        "mono_viol_mag_max":     phys["mono_viol_magnitude_max"],
        "neg_rate":              phys["neg_rate"],
        "PC_R2":                 round(pc_r2, 6),
    }


def macro_metrics(
    per_client: List[Dict[str, float]],
) -> Dict[str, float]:
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
    """Return per-client R2, RMSE, PVR, neg_rate as flat dict for logging."""
    out: Dict[str, float] = {}
    for i, m in enumerate(per_client):
        out[f"c{i}_R2"]   = m["R2"]
        out[f"c{i}_RMSE"] = m["RMSE"]
        out[f"c{i}_PVR"]  = m["mono_viol_rate"]
        out[f"c{i}_neg"]  = m["neg_rate"]
        out[f"c{i}_PC_R2"] = m["PC_R2"]
    return out


# ── Gate scoring ──────────────────────────────────────────────────────────────

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
    tmp = ElectrolyzerMLP(use_hard_guard=use_hard_guard).to(device)
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


# ── Trust gate decision ───────────────────────────────────────────────────────

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
            "candidate_metrics must contain 'R2' or 'R2_macro'. "
            f"Got keys: {sorted(candidate_metrics.keys())}")

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


# ── Physics forgetting ────────────────────────────────────────────────────────

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
        tmp = ElectrolyzerMLP(use_hard_guard=use_hard_guard).to(device)
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


def aggregate_forgetting(
    pf_records: List[Dict[str, float]],
) -> Dict[str, float]:
    if not pf_records:
        return {"apf_signed": 0.0, "apf_positive": 0.0,
                "pf_peak": 0.0, "n_forgetting": 0, "n_total_steps": 0}
    signed   = [r["pf_signed"]   for r in pf_records]
    positive = [r["pf_positive"] for r in pf_records]
    return {
        "apf_signed":   float(np.mean(signed)),
        "apf_positive": float(np.mean(positive)),
        "pf_peak":      float(np.max(positive)),
        "n_forgetting": int(sum(1 for s in signed if s > 0)),
        "n_total_steps": len(pf_records),
    }


# ── CWC utilities ─────────────────────────────────────────────────────────────

def compute_fisher_diagonal(
    model:     ElectrolyzerMLP,
    X_norm:    np.ndarray,
    y_norm:    np.ndarray,
    I_phys:    np.ndarray,
    std:       Standardizer,
    device:    torch.device,
    n_samples: int = 200,
    seed:      int = 0,
) -> Dict[str, torch.Tensor]:
    model.eval()
    rng = np.random.default_rng(seed)
    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param)
        for name, param in model.body.named_parameters()
    }
    N   = len(X_norm)
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    for i in idx:
        model.zero_grad()
        It     = torch.tensor(I_phys[i:i+1], dtype=torch.float32).to(device)
        Xt     = torch.tensor(X_norm[i:i+1], dtype=torch.float32).to(device)
        yt     = torch.tensor(y_norm[i:i+1], dtype=torch.float32).to(device)
        y_pred = model(It, Xt, I_ref=std.I_mean, H_ref=std.y_mean, y_std=std.y_std)
        F.mse_loss(y_pred, yt).backward()
        for name, param in model.body.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
    for name in fisher:
        fisher[name] /= min(n_samples, N)
    return fisher


def cwc_penalty(
    model:          ElectrolyzerMLP,
    ref_body_state: Dict[str, torch.Tensor],
    fisher:         Dict[str, torch.Tensor],
    device:         torch.device,
) -> torch.Tensor:
    penalty    = torch.tensor(0.0, device=device, requires_grad=False)
    body_params = dict(model.body.named_parameters())
    for name, ref_val in ref_body_state.items():
        short = name.replace("body.", "")
        if short in body_params and short in fisher:
            diff    = body_params[short] - ref_val.to(device)
            penalty = penalty + (fisher[short].to(device) * diff.pow(2)).sum()
    return penalty


# ── Result I/O ────────────────────────────────────────────────────────────────

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


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(
    name:     str,
    log_file: Optional[str] = None,
    level:    int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── Print utilities ───────────────────────────────────────────────────────────

def separator(title: str = "", width: int = 70) -> None:
    line = "=" * width
    print(f"\n{line}\n  {title}\n{line}" if title else line)


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {prefix}{k:<30s}: {v:.6f}")
        else:
            print(f"  {prefix}{k:<30s}: {v}")
