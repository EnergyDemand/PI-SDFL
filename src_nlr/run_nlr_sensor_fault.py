"""
run_nlr_sensor_fault.py — NLR input-space sensor fault experiment.

Applies a gain error to the current input of Client 4 (simulated_tidal).
The measured H2 target is unchanged.

Gain levels evaluated: 1.1 (primary), 1.2 (secondary).

Methods run at each gain level:
    PI-SDFL-full
    Vanilla-Sequential
    PI-Vanilla-Sequential
    FedAvg

Outputs saved under results_nlr/sensor_fault/gain_<gXXX>/<Method>/

Usage:
    python run_nlr_sensor_fault.py --mode smoke
    python run_nlr_sensor_fault.py --mode pilot
    python run_nlr_sensor_fault.py --mode final
    python run_nlr_sensor_fault.py --mode final --gains 1.1
"""

import argparse
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nlr as C
from corrupt_utils_nlr import apply_sensor_fault, print_fault_summary, PRIMARY_GAIN
from experiment_utils_nlr import (
    ExperimentConfig, make_config, load_data, load_probe, build_split_manifest,
    ClientSplitNLR, DEVICE, RESULTS_DIR,
    new_model, mu_schedule, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, print_result_line,
)
from run_nlr_pi_sdfl          import run_single as run_pi_sdfl
from run_nlr_vanilla_sequential import run      as run_vanilla
from run_nlr_pi_vanilla_sequential import run   as run_pi_vanilla
from run_nlr_fedavg           import run        as run_fedavg

DEFAULT_GAINS = [1.1, 1.2]

METHODS = {
    "PI-SDFL-full":         run_pi_sdfl,
    "Vanilla-Sequential":   run_vanilla,
    "PI-Vanilla-Sequential":run_pi_vanilla,
    "FedAvg":               run_fedavg,
}

# PI-SDFL has use_hard_guard=True; others use False.
USE_PI = {
    "PI-SDFL-full":          True,
    "Vanilla-Sequential":    False,
    "PI-Vanilla-Sequential": True,
    "FedAvg":                False,
}


def gain_tag(gain: float) -> str:
    return f"gain_{int(round(gain * 100)):03d}"


def run_one_method_one_gain(
    method_name: str,
    run_fn,
    gain:        float,
    splits_clean:List[ClientSplitNLR],
    std:         C.Standardizer,
    cfg:         ExperimentConfig,
    seed:        int,
    X_eval_norm: np.ndarray,
    y_eval_raw:  np.ndarray,
    I_eval_phys: np.ndarray,
    I_probe:     torch.Tensor,
    x_probe:     torch.Tensor,
    out_dir:     str,
    manifest:    Dict,
) -> Dict:
    corrupted_splits, diag = apply_sensor_fault(splits_clean, gain=gain, std=std)
    print_fault_summary(diag)

    if method_name == "PI-SDFL-full":
        result = run_fn(
            splits=corrupted_splits, std=std, cfg=cfg, seed=seed,
            X_eval_norm=X_eval_norm, y_eval_raw=y_eval_raw,
            I_eval_phys=I_eval_phys, I_probe=I_probe, x_probe=x_probe,
            verbose=False,
        )
        m, rl, gl, pfl, al, gsl, cvl, pcl, bb, fb, hd = result
    else:
        result = run_fn(
            splits=corrupted_splits, std=std, cfg=cfg, seed=seed,
            X_eval_norm=X_eval_norm, y_eval_raw=y_eval_raw,
            I_eval_phys=I_eval_phys, I_probe=I_probe, x_probe=x_probe,
        )
        m, rl, gl, pfl, pcl, cvl, bb, fb, hd = result
        al  = []
        gsl = []

    m["sensor_fault_gain"]  = float(gain)
    m["corrupted_client"]   = 4
    m["target_modified"]    = False

    manifest_with_fault = {**manifest, "sensor_fault": diag}

    save_run(out_dir, method_name, seed, cfg, m, rl,
             gate_log=gl, pf_log=pfl,
             activity_log=(al if al else None),
             gate_scores_log=(gsl if gsl else None),
             per_client_log=pcl,
             convergence_log=(cvl if cvl else None),
             best_body=bb, final_body=fb,
             personalized_heads=hd, std=std,
             split_manifest=manifest_with_fault)
    print_result_line(method_name, m, cfg,
                      gate=(method_name == "PI-SDFL-full"))
    return m


def main(mode: str, gains: List[float]) -> None:
    cfg = make_config(mode)

    print(f"\n{'='*60}")
    print(f"  NLR Sensor Fault  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(cfg.seeds)}  rounds={cfg.rounds}")
    print(f"  gains={gains}")
    print(f"{'='*60}\n")

    for gain in gains:
        tag     = gain_tag(gain)
        out_dir = os.path.join(RESULTS_DIR, "sensor_fault", tag)
        os.makedirs(out_dir, exist_ok=True)

        agg = {name: [] for name in METHODS}

        for seed in cfg.seeds:
            print(f"\n  GAIN={gain:.2f}  SEED={seed}")
            splits_clean, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(cfg)
            I_probe, x_probe = load_probe(DEVICE)
            manifest = build_split_manifest(cfg, splits_clean)

            for method_name, run_fn in METHODS.items():
                print(f"  [{method_name}]...", end=" ", flush=True)
                C.set_seed(seed)

                m = run_one_method_one_gain(
                    method_name=method_name,
                    run_fn=run_fn,
                    gain=gain,
                    splits_clean=splits_clean,
                    std=std,
                    cfg=cfg,
                    seed=seed,
                    X_eval_norm=X_eval_norm,
                    y_eval_raw=y_eval_raw,
                    I_eval_phys=I_eval_phys,
                    I_probe=I_probe,
                    x_probe=x_probe,
                    out_dir=out_dir,
                    manifest=manifest,
                )
                agg[method_name].append(m)

        aggregate_and_save_summary(agg, out_dir)
        print(f"\n  Sensor fault gain={gain:.2f} results -> {out_dir}")

    print(f"\nSensor fault experiment COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLR sensor fault experiment (input-space gain error on Client 4)")
    parser.add_argument(
        "--mode", type=str, default="smoke",
        choices=["smoke", "pilot", "final"],
    )
    parser.add_argument(
        "--gains", nargs="+", type=float, default=DEFAULT_GAINS,
        help="Gain values to evaluate (default: 1.1 1.2)",
    )
    args = parser.parse_args()
    main(args.mode, args.gains)
