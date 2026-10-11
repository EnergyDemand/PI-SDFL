"""
run_nlr_local_only.py — Local-only reference for NLR experiment.

Each client trains independently for the equivalent total epochs.
No shared body. Global R2 metrics are None (no shared model to evaluate).

Usage:
    python run_nlr_local_only.py --mode smoke
    python run_nlr_local_only.py --mode pilot
    python run_nlr_local_only.py --mode final
"""

import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nlr as C
from experiment_utils_nlr import (
    ExperimentConfig, make_config, load_data, load_probe, build_split_manifest,
    ClientSplitNLR, DEVICE, RESULTS_DIR,
    new_model, recalib_head, get_heads,
    eval_per_client_personalised,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)

METHOD  = "Local-only"
OUT_DIR = os.path.join(RESULTS_DIR, "clean")


def run(
    splits:      List[ClientSplitNLR],
    std:         C.Standardizer,
    cfg:         ExperimentConfig,
    seed:        int,
    X_eval_norm: np.ndarray,
    y_eval_raw:  np.ndarray,
    I_eval_phys: np.ndarray,
    I_probe:     torch.Tensor,
    x_probe:     torch.Tensor,
) -> tuple:
    C.set_seed(seed)
    total_ep = cfg.pretrain_ep + cfg.rounds * cfg.local_epochs
    use_pi   = False
    models   = [new_model(use_pi) for _ in splits]

    for i, (m, cd) in enumerate(zip(models, splits)):
        C.set_seed(seed + i * 1000)
        C.train_local(
            model=m, X_norm=cd.X_train, y_norm=cd.y_train, I_phys=cd.I_train,
            n_epochs=total_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed + i * 1000, use_mono_loss=False, beta_mono=0.0,
            use_hard_guard=use_pi, device=DEVICE, std=std,
        )

    client_flat, per_client = eval_per_client_personalised(
        models, splits, std, I_probe, x_probe)

    per_client_log = [
        {"seed": seed, "client": i, **pm}
        for i, pm in enumerate(per_client)
    ]

    metrics = dict(client_flat)
    for k in ["global_R2", "global_RMSE", "global_MAE",
              "global_mono_viol_rate", "global_mono_viol_magnitude",
              "global_mono_viol_mag_max", "global_neg_rate", "global_PC_R2"]:
        metrics[k] = None
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds

    heads = get_heads(models, body_state=None)
    return metrics, [], [], [], per_client_log, None, None, heads


def main(mode: str) -> None:
    cfg = make_config(mode)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {METHOD}  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(cfg.seeds)}  rounds={cfg.rounds}")
    print(f"{'='*60}\n")

    agg = {METHOD: []}

    for seed in cfg.seeds:
        print(f"  SEED {seed}")
        splits, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(cfg)
        I_probe, x_probe = load_probe(DEVICE)
        manifest = build_split_manifest(cfg, splits)

        C.set_seed(seed)
        m, rl, gl, pfl, pcl, bb, fb, hd = run(
            splits, std, cfg, seed, X_eval_norm, y_eval_raw,
            I_eval_phys, I_probe, x_probe)
        agg[METHOD].append(m)

        save_run(OUT_DIR, METHOD, seed, cfg, m, rl,
                 gate_log=gl, pf_log=pfl,
                 per_client_log=pcl,
                 best_body=bb, final_body=fb,
                 personalized_heads=hd, std=std, split_manifest=manifest)
        print_result_line(METHOD, m, cfg)

    aggregate_and_save_summary(agg, OUT_DIR)
    print(f"\nResults saved -> {OUT_DIR}")


if __name__ == "__main__":
    parser = base_arg_parser("NLR Local-only reference")
    args   = parser.parse_args()
    main(args.mode)
