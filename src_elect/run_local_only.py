"""
run_local_only.py — Local-only reference bound.

Each client trains independently for the equivalent total epochs
(pretrain + rounds * local_epochs). No shared body. Lower bound on
federated performance.

Usage:
    python run_local_only.py --mode smoke
    python run_local_only.py --mode pilot
    python run_local_only.py --mode final
"""

import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from experiment_utils import (
    ExperimentConfig, make_config, load_data, build_split_manifest,
    ClientSplit, DEVICE, RESULTS_DIR,
    new_model, recalib_head, get_heads,
    eval_global, eval_per_client_personalised, global_eval_model,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)


METHOD = "Local-only"


def run(
    splits:      List[ClientSplit],
    std:         C.Standardizer,
    cfg:         ExperimentConfig,
    seed:        int,
    X_eval_norm: np.ndarray,
    y_eval_raw:  np.ndarray,
    I_eval_phys: np.ndarray,
    I_probe:     torch.Tensor,
    x_probe:     torch.Tensor,
) -> Dict:
    C.set_seed(seed)
    total_ep = cfg.pretrain_ep + cfg.rounds * cfg.local_epochs
    use_pi   = False
    models   = [new_model(use_pi) for _ in splits]

    for i, (m, cd) in enumerate(zip(models, splits)):
        C.set_seed(seed + i * 1000)
        C.train_local(
            model=m, X_norm=cd.X_train,
            y_norm=std.transform_y(cd.y_train), I_phys=cd.I_train,
            n_epochs=total_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed + i * 1000, use_mono_loss=False, beta_mono=0.0,
            use_hard_guard=use_pi, device=DEVICE, std=std,
        )

    client_flat, per_client = eval_per_client_personalised(
        models, splits, std, I_probe, x_probe)

    per_client_log = [
        {"seed": seed, "client": i, **{k: v for k, v in m.items()}}
        for i, m in enumerate(per_client)
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
    cfg      = make_config(mode)
    run_name = mode
    out_dir  = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    eval_file = "global_test.csv" if mode == "final" else "global_val.csv"

    print(f"\n{'='*60}")
    print(f"  {METHOD}  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(cfg.seeds)}  rounds={cfg.rounds}")
    print(f"{'='*60}\n")

    agg = {METHOD: []}

    for seed in cfg.seeds:
        print(f"  SEED {seed}")
        splits, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(
            cfg, eval_file=eval_file, split_seed=0)
        manifest = build_split_manifest(cfg, splits, eval_file)

        I_probe, x_probe = C.build_probe_grid(std, DEVICE)

        C.set_seed(seed)
        result = run(splits, std, cfg, seed, X_eval_norm, y_eval_raw,
                     I_eval_phys, I_probe, x_probe)
        m, rl, gl, pfl, pcl, bb, fb, hd = result
        agg[METHOD].append(m)

        save_run(
            out_dir, METHOD, seed, cfg, m, rl,
            gate_log=gl, pf_log=pfl,
            per_client_log=pcl,
            best_body=bb, final_body=fb,
            personalized_heads=hd, std=std, split_manifest=manifest,
        )
        print_result_line(METHOD, m, cfg)

    aggregate_and_save_summary(agg, out_dir)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    parser = base_arg_parser("Local-only reference bound")
    args   = parser.parse_args()
    main(args.mode)
