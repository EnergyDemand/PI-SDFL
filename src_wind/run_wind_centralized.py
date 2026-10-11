"""
run_wind_centralized.py — Centralized reference for wind AEI experiment.

Usage:
    python run_wind_centralized.py --mode smoke
    python run_wind_centralized.py --mode pilot
    python run_wind_centralized.py --mode final
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_wind_aei as C
from experiment_utils_wind import (
    ExperimentConfig, make_config, load_data, load_probe, build_split_manifest,
    ClientSplitWind, DEVICE, RESULTS_DIR,
    new_model, get_heads, global_eval_model,
    eval_global, eval_shared_model_per_client,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)
METHOD  = "Centralized"
OUT_DIR = os.path.join(RESULTS_DIR, "clean")

def run(splits, std, cfg, seed, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe):
    C.set_seed(seed)
    use_pi   = False
    total_ep = cfg.pretrain_ep + cfg.rounds * cfg.local_epochs
    X_pool = np.vstack([cd.X_train for cd in splits])
    y_pool = np.concatenate([cd.y_train for cd in splits])
    I_pool = np.concatenate([cd.I_train for cd in splits])
    m = new_model(use_pi)
    C.train_local(model=m, X_norm=X_pool, y_norm=y_pool, I_phys=I_pool,
                  n_epochs=total_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                  seed=seed, use_mono_loss=False, beta_mono=0.0,
                  use_hard_guard=use_pi, device=DEVICE, std=std)
    round_log  = [{"seed": seed, "round": r + 1} for r in range(cfg.rounds)]
    final_body = m.get_body_state()

    # Evaluate using the same pooled-calibration head protocol as all
    # other shared-body methods (Vanilla, FedAvg, CWC, PI-SDFL, etc.)
    eval_m = global_eval_model(final_body, splits, std, cfg, seed, use_pi)
    gm = eval_global(eval_m, std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe)
    client_flat, per_client = eval_shared_model_per_client(eval_m, splits, std, I_probe, x_probe)

    per_client_log = [{"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]
    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds
    return metrics, round_log, [], [], per_client_log, None, final_body, get_heads([eval_m], body_state=final_body)

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
        m, rl, gl, pfl, pcl, bb, fb, hd = run(splits, std, cfg, seed, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe)
        agg[METHOD].append(m)
        save_run(OUT_DIR, METHOD, seed, cfg, m, rl, gate_log=gl, pf_log=pfl, per_client_log=pcl, best_body=bb, final_body=fb, personalized_heads=hd, std=std, split_manifest=manifest)
        print_result_line(METHOD, m, cfg)
    aggregate_and_save_summary(agg, OUT_DIR)
    print(f"\nResults saved -> {OUT_DIR}")

if __name__ == "__main__":
    args = base_arg_parser("Wind AEI Centralized reference").parse_args()
    main(args.mode)
