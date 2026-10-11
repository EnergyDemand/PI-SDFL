"""
run_pi_vanilla_sequential.py — PI-Vanilla-Sequential baseline.

PI architecture in a sequential ring without gate, anchor, or rollback.
L_mono and hard guard are active throughout (pretrain AND ring training).

Fix applied (AEI revision):
    Ring training loop previously passed use_mono_loss=False, beta_mono=0.0,
    silently disabling L_mono after pretraining. Corrected to use cfg values
    and pass I_probe / x_probe as required by train_local().

Usage:
    python run_pi_vanilla_sequential.py --mode smoke
    python run_pi_vanilla_sequential.py --mode pilot
    python run_pi_vanilla_sequential.py --mode final
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from experiment_utils import (
    ExperimentConfig, make_config, load_data, build_split_manifest,
    ClientSplit, DEVICE, RESULTS_DIR,
    new_model, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)


METHOD = "PI-Vanilla-Sequential"


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
    init_body:   Optional[Dict] = None,
) -> Dict:
    C.set_seed(seed)
    use_pi = True
    n      = len(splits)
    models = [new_model(use_pi) for _ in range(n)]

    if init_body is not None:
        start_body = init_body
    elif cfg.pretrain_ep > 0:
        C.train_local(
            model=models[0], X_norm=splits[0].X_train,
            y_norm=std.transform_y(splits[0].y_train), I_phys=splits[0].I_train,
            n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed,
            use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
            use_hard_guard=use_pi, device=DEVICE, std=std,
            I_probe_phys=(I_probe if cfg.use_mono_loss else None),
            x_probe_norm=(x_probe if cfg.use_mono_loss else None),
        )
        start_body = models[0].get_body_state()
    else:
        start_body = models[0].get_body_state()

    for m in models:
        m.set_body_state(start_body)
        m.reset_head(0.0)

    cyc_body        = C.clone_state(start_body)
    round_log       = []
    convergence_log = []

    for r in range(cfg.rounds):
        pvr_before_round = []
        pvr_after_round  = []

        for i, (m, cd) in enumerate(zip(models, splits)):
            C.set_seed(seed + r * 100 + i + 1)
            m.set_body_state(cyc_body)

            pm_before = C.physics_metrics(m, I_probe, x_probe, std, DEVICE)
            C.train_local(
                model=m, X_norm=cd.X_train,
                y_norm=std.transform_y(cd.y_train), I_phys=cd.I_train,
                n_epochs=cfg.local_epochs, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                seed=seed + r * 100 + i + 1,
                use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
                use_hard_guard=use_pi, device=DEVICE, std=std,
                I_probe_phys=(I_probe if cfg.use_mono_loss else None),
                x_probe_norm=(x_probe if cfg.use_mono_loss else None),
            )
            pm_after = C.physics_metrics(m, I_probe, x_probe, std, DEVICE)

            cyc_body = m.get_body_state()
            pvr_before_round.append(pm_before["mono_viol_rate"])
            pvr_after_round.append(pm_after["mono_viol_rate"])

        round_log.append({
            "seed":            seed,
            "round":           r + 1,
            "pvr_mean_before": float(np.mean(pvr_before_round)),
            "pvr_mean_after":  float(np.mean(pvr_after_round)),
        })

        tmp = new_model(use_pi)
        tmp.set_body_state(cyc_body)
        convergence_log.extend(
            eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

    final_body = cyc_body
    for i, (m, cd) in enumerate(zip(models, splits)):
        m.set_body_state(final_body)
        m.reset_head(0.0)
        recalib_head(m, cd, std, cfg, seed + 8000 + i, use_pi)

    gm = eval_global(
        global_eval_model(final_body, splits, std, cfg, seed, use_pi),
        std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe,
    )
    client_flat, per_client = eval_per_client_personalised(
        models, splits, std, I_probe, x_probe)

    per_client_log = [
        {"seed": seed, "client": i, **{k: v for k, v in pm.items()}}
        for i, pm in enumerate(per_client)
    ]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds

    heads = get_heads(models, body_state=final_body)
    return metrics, round_log, [], [], per_client_log, convergence_log, None, final_body, heads


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
        m, rl, gl, pfl, pcl, cvl, bb, fb, hd = result
        agg[METHOD].append(m)

        save_run(
            out_dir, METHOD, seed, cfg, m, rl,
            gate_log=gl, pf_log=pfl,
            per_client_log=pcl, convergence_log=cvl,
            best_body=bb, final_body=fb,
            personalized_heads=hd, std=std, split_manifest=manifest,
        )
        print_result_line(METHOD, m, cfg)

    aggregate_and_save_summary(agg, out_dir)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    parser = base_arg_parser("PI-Vanilla-Sequential baseline (L_mono active in ring)")
    args   = parser.parse_args()
    main(args.mode)
