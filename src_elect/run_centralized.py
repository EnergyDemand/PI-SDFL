"""
run_centralized.py — Centralized reference.

All client training data pooled into one dataset. Star topology.
Reference baseline — not a guaranteed upper bound on federated performance.

Usage:
    python run_centralized.py --mode smoke
    python run_centralized.py --mode pilot
    python run_centralized.py --mode final
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
    new_model, get_heads,
    eval_global, eval_shared_model_per_client, global_eval_model,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)


METHOD = "Centralized"


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
    use_pi   = False
    total_ep = cfg.pretrain_ep + cfg.rounds * cfg.local_epochs

    X_pool = np.vstack([cd.X_train for cd in splits])
    y_pool = np.concatenate([cd.y_train for cd in splits])
    I_pool = np.concatenate([cd.I_train for cd in splits])

    m = new_model(use_pi)
    C.train_local(
        model=m, X_norm=X_pool, y_norm=std.transform_y(y_pool), I_phys=I_pool,
        n_epochs=total_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head, seed=seed,
        use_mono_loss=False, beta_mono=0.0,
        use_hard_guard=use_pi, device=DEVICE, std=std,
    )

    round_log  = [{"seed": seed, "round": r + 1} for r in range(cfg.rounds)]
    final_body = m.get_body_state()

    # Evaluate using the same pooled-calibration head protocol as all
    # other shared-body methods (Vanilla, FedAvg, CWC, PI-SDFL, etc.)
    eval_m = global_eval_model(final_body, splits, std, cfg, seed, use_pi)
    metrics = eval_global(eval_m, std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe)

    client_flat, per_client = eval_shared_model_per_client(eval_m, splits, std, I_probe, x_probe)
    metrics.update(client_flat)

    per_client_log = [
        {"seed": seed, "client": i, **{k: v for k, v in pm.items()}}
        for i, pm in enumerate(per_client)
    ]

    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds

    heads = get_heads([eval_m], body_state=final_body)
    return metrics, round_log, [], [], per_client_log, None, final_body, heads


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
    parser = base_arg_parser("Centralized reference baseline")
    args   = parser.parse_args()
    main(args.mode)
