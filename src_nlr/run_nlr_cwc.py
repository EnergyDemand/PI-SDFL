"""
run_nlr_cwc.py — CWC baseline for NLR experiment.

Sequential ring with Fisher-weighted EWC-style regularisation.
Generic MLP, no physics architecture.

Usage:
    python run_nlr_cwc.py --mode smoke
    python run_nlr_cwc.py --mode pilot
    python run_nlr_cwc.py --mode final
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nlr as C
from experiment_utils_nlr import (
    ExperimentConfig, make_config, load_data, load_probe, build_split_manifest,
    ClientSplitNLR, DEVICE, RESULTS_DIR,
    new_model, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)

METHOD  = "CWC"
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
    use_pi = False
    n      = len(splits)
    models = [new_model(use_pi) for _ in range(n)]

    if cfg.pretrain_ep > 0:
        C.train_local(
            model=models[0], X_norm=splits[0].X_train,
            y_norm=splits[0].y_train, I_phys=splits[0].I_train,
            n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed, use_mono_loss=False, beta_mono=0.0,
            use_hard_guard=use_pi, device=DEVICE, std=std,
        )
    start_body = models[0].get_body_state()

    for m in models:
        m.set_body_state(start_body)
        m.reset_head(0.0)

    cyc_body        = C.clone_state(start_body)
    ref_body        = None
    fisher_ref      = None
    round_log       = []
    convergence_log = []

    for r in range(cfg.rounds):
        pvr_before_round, pvr_after_round = [], []

        for i, (m, cd) in enumerate(zip(models, splits)):
            C.set_seed(seed + r * 100 + i + 1)
            m.set_body_state(cyc_body)

            pm_before = C.physics_metrics(m, I_probe, x_probe, std, DEVICE)

            if ref_body is not None and fisher_ref is not None:
                ref_dev  = {k: v.to(DEVICE).detach() for k, v in ref_body.items()}
                fish_dev = {k: v.to(DEVICE).detach() for k, v in fisher_ref.items()}
                opt = torch.optim.Adam([
                    {"params": m.body.parameters(), "lr": cfg.lr_body},
                    {"params": [m.alpha_k],          "lr": cfg.lr_head},
                ])
                It = torch.tensor(cd.I_train, dtype=torch.float32)
                Xt = torch.tensor(cd.X_train, dtype=torch.float32)
                yt = torch.tensor(cd.y_train, dtype=torch.float32)
                dataset = torch.utils.data.TensorDataset(It, Xt, yt)
                loader  = torch.utils.data.DataLoader(
                    dataset, batch_size=C.BATCH_SIZE, shuffle=True,
                    generator=C.make_dataloader_generator(seed + r * 100 + i + 1),
                    drop_last=False,
                )
                for ep in range(cfg.local_epochs):
                    batches   = list(loader)
                    n_batches = len(batches)
                    for b_idx, (I_b, X_b, y_b) in enumerate(batches):
                        I_b, X_b, y_b = I_b.to(DEVICE), X_b.to(DEVICE), y_b.to(DEVICE)
                        opt.zero_grad()
                        y_pred = m(I_b, X_b,
                                   I_ref=std.I_mean,
                                   H_ref=std.y_mean,
                                   y_std=std.y_std)
                        l_data = F.mse_loss(y_pred, y_b)
                        if b_idx == n_batches - 1:
                            cwc_pen = C.cwc_penalty(m, ref_dev, fish_dev, DEVICE)
                            (l_data + cfg.cwc_lambda * cwc_pen).backward()
                        else:
                            l_data.backward()
                        opt.step()
            else:
                C.train_local(
                    model=m, X_norm=cd.X_train, y_norm=cd.y_train,
                    I_phys=cd.I_train, n_epochs=cfg.local_epochs,
                    lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                    seed=seed + r * 100 + i + 1,
                    use_mono_loss=False, beta_mono=0.0,
                    use_hard_guard=use_pi, device=DEVICE, std=std,
                )

            pm_after = C.physics_metrics(m, I_probe, x_probe, std, DEVICE)

            cyc_body   = m.get_body_state()
            ref_body   = C.clone_state(cyc_body)
            fisher_ref = C.compute_fisher_diagonal(
                m, cd.X_train, cd.y_train, cd.I_train, std, DEVICE,
                n_samples=100, seed=seed + r * 100 + i + 1,
            )

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
        {"seed": seed, "client": i, **pm}
        for i, pm in enumerate(per_client)
    ]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds

    heads = get_heads(models, body_state=final_body)
    return metrics, round_log, [], [], per_client_log, convergence_log, None, final_body, heads


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
        m, rl, gl, pfl, pcl, cvl, bb, fb, hd = run(
            splits, std, cfg, seed, X_eval_norm, y_eval_raw,
            I_eval_phys, I_probe, x_probe)
        agg[METHOD].append(m)

        save_run(OUT_DIR, METHOD, seed, cfg, m, rl,
                 gate_log=gl, pf_log=pfl,
                 per_client_log=pcl, convergence_log=cvl,
                 best_body=bb, final_body=fb,
                 personalized_heads=hd, std=std, split_manifest=manifest)
        print_result_line(METHOD, m, cfg)

    aggregate_and_save_summary(agg, OUT_DIR)
    print(f"\nResults saved -> {OUT_DIR}")


if __name__ == "__main__":
    parser = base_arg_parser("NLR CWC sequential baseline")
    args   = parser.parse_args()
    main(args.mode)
