"""
run_wind_frozen_reference.py — Frozen Reference control for wind AEI experiment.

Same role as run_frozen_reference.py in src_elect_new:
Builds the PI warm-start body from Client 0 (WT3), freezes it entirely,
and evaluates without any ring training. Answers the question of whether
PI-SDFL's gate-protection fallback is distinguishable from simply freezing.

Usage:
    python run_wind_frozen_reference.py --mode smoke
    python run_wind_frozen_reference.py --mode pilot
    python run_wind_frozen_reference.py --mode final
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_wind_aei as C
from experiment_utils_wind import (
    ExperimentConfig, make_config, load_data, load_probe, build_split_manifest,
    ClientSplitWind, DEVICE, RESULTS_DIR,
    new_model, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)

METHOD  = "Frozen-Reference"
OUT_DIR = os.path.join(RESULTS_DIR, "clean")


def run(splits, std, cfg, seed, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe):
    C.set_seed(seed)
    use_pi      = True
    warm_model  = new_model(use_pi)

    if cfg.pretrain_ep > 0:
        C.train_local(model=warm_model, X_norm=splits[0].X_train,
                      y_norm=splits[0].y_train, I_phys=splits[0].I_train,
                      n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                      seed=seed,
                      use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
                      use_hard_guard=use_pi, device=DEVICE, std=std,
                      I_probe_phys=(I_probe if cfg.use_mono_loss else None),
                      x_probe_norm=(x_probe if cfg.use_mono_loss else None))

    frozen_body     = warm_model.get_body_state()
    round_log       = []
    convergence_log = []

    for r in range(cfg.rounds):
        round_log.append({"seed": seed, "round": r + 1, "body_frozen": True, "n_updates": 0})
        tmp = new_model(use_pi)
        tmp.set_body_state(frozen_body)
        convergence_log.extend(eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

    final_body = frozen_body
    models = [new_model(use_pi) for _ in splits]
    for i, (m, cd) in enumerate(zip(models, splits)):
        m.set_body_state(final_body)
        m.reset_head(0.0)
        recalib_head(m, cd, std, cfg, seed + 8000 + i, use_pi)

    gm = eval_global(global_eval_model(final_body, splits, std, cfg, seed, use_pi),
                     std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe)
    client_flat, per_client = eval_per_client_personalised(models, splits, std, I_probe, x_probe)
    per_client_log = [{"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_updates"]  = 0
    metrics["n_rounds"]   = cfg.rounds
    return metrics, round_log, [], [], per_client_log, convergence_log, None, final_body, get_heads(models, body_state=final_body)


def main(mode):
    cfg = make_config(mode)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*60}\n  {METHOD}  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(cfg.seeds)}  rounds={cfg.rounds}\n{'='*60}\n")
    agg = {METHOD: []}
    for seed in cfg.seeds:
        print(f"  SEED {seed}")
        splits, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(cfg)
        I_probe, x_probe = load_probe(DEVICE)
        manifest = build_split_manifest(cfg, splits)
        C.set_seed(seed)
        m, rl, gl, pfl, pcl, cvl, bb, fb, hd = run(splits, std, cfg, seed, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe)
        agg[METHOD].append(m)
        save_run(OUT_DIR, METHOD, seed, cfg, m, rl, gate_log=gl, pf_log=pfl, per_client_log=pcl, convergence_log=cvl, best_body=bb, final_body=fb, personalized_heads=hd, std=std, split_manifest=manifest)
        print_result_line(METHOD, m, cfg)
    aggregate_and_save_summary(agg, OUT_DIR)
    print(f"\nResults saved -> {OUT_DIR}")


if __name__ == "__main__":
    args = base_arg_parser("Wind AEI Frozen Reference control").parse_args()
    main(args.mode)
