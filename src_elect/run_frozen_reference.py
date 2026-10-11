"""
run_frozen_reference.py — Frozen Reference control for PI-SDFL AEI experiments.

Purpose:
    Directly answers the ESWA reviewer criticism that PI-SDFL under severe
    corruption (0/20 accepted) is indistinguishable from simply freezing the
    warm-start body. This runner makes that frozen baseline explicit and
    comparable in the results tables.

Protocol:
    1. Build the same PI warm-start body as PI-SDFL (Client 1, pretrain_ep,
       hard guard + L_mono active).
    2. Freeze the body entirely — no ring training, no updates of any kind.
    3. Log cfg.rounds identical no-update entries for plotting continuity.
    4. Apply the same final pooled calibration-head procedure as all other methods.
    5. Evaluate on the same global eval file.

Output structure is identical to all other method runners so it enters the
same aggregate_and_save_summary pipeline cleanly.

Usage:
    python run_frozen_reference.py --mode smoke
    python run_frozen_reference.py --mode pilot
    python run_frozen_reference.py --mode final
"""

import os
import sys
from typing import Dict, List, Tuple

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


METHOD = "Frozen-Reference"


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
) -> Tuple:
    C.set_seed(seed)
    use_pi = True

    warm_model = new_model(use_pi)
    if cfg.pretrain_ep > 0:
        C.train_local(
            model=warm_model, X_norm=splits[0].X_train,
            y_norm=std.transform_y(splits[0].y_train), I_phys=splits[0].I_train,
            n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed,
            use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
            use_hard_guard=use_pi, device=DEVICE, std=std,
            I_probe_phys=(I_probe if cfg.use_mono_loss else None),
            x_probe_norm=(x_probe if cfg.use_mono_loss else None),
        )

    frozen_body = warm_model.get_body_state()

    round_log       = []
    convergence_log = []

    for r in range(cfg.rounds):
        round_log.append({
            "seed":        seed,
            "round":       r + 1,
            "body_frozen": True,
            "n_updates":   0,
        })

        tmp = new_model(use_pi)
        tmp.set_body_state(frozen_body)
        convergence_log.extend(
            eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

    final_body = frozen_body

    models = [new_model(use_pi) for _ in splits]
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
    metrics["n_updates"]  = 0
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
    parser = base_arg_parser("Frozen Reference control — warm-start body, no ring updates")
    args   = parser.parse_args()
    main(args.mode)
