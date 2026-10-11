"""
run_severity_sweep.py — Fault severity sweep for PI-SDFL AEI experiments.

Purpose:
    Demonstrates adaptive gate behaviour across a range of fault severities.
    Replaces the single extreme-case (frac=1.00) from the ESWA submission
    with a full severity profile showing:
        Clean      -> nearly all rounds accepted
        Mild       -> selective acceptance begins
        Moderate   -> clear selective rejection
        Strong     -> mostly protective
        Severe     -> safe freeze (equivalent to Frozen Reference)

Corruption model:
    Current-conditioned H2 target corruption (Client 4 y_train only).
    quantile q=0.80 fixed throughout. frac controls severity:
        0.00 -> no corruption (clean reference via corruption path)
        0.25 -> mild
        0.35 -> intermediate
        0.50 -> strong
        0.75 -> near-severe
        1.00 -> full inversion (severe, paper original)

    Note: I_train is never modified. This is a target-space corruption
    conditioned on high-current samples, not a true input-sensor fault.
    Label accordingly in the manuscript.

Methods run at each severity level:
    Frozen-Reference        (control — body never updates)
    Vanilla-Sequential      (no physics, no gate)
    PI-Vanilla-Sequential   (physics, no gate — corrected L_mono)
    PI-SDFL-full            (proposed method)

Output layout:
    results_new/severity_sweep/frac_XXX/
        summary.json and per-method/per-seed folders matching standard format.

Usage:
    python run_severity_sweep.py --mode pilot
    python run_severity_sweep.py --mode final
    python run_severity_sweep.py --mode pilot --fracs 0.00 0.25 0.50 1.00
"""

import os
import sys
import argparse
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import corrupt_utils as CU
from experiment_utils import (
    ExperimentConfig, make_config, load_data, build_split_manifest,
    ClientSplit, DEVICE, RESULTS_DIR,
    new_model, mu_schedule, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, base_arg_parser, print_result_line,
)

DEFAULT_FRACS    = [0.00, 0.25, 0.35, 0.50, 0.75, 1.00]
FIXED_QUANTILE   = CU.DEFAULT_QUANTILE


# ── Frozen Reference ──────────────────────────────────────────────────────────

def run_frozen_reference(
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
    use_pi     = True
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

    frozen_body     = warm_model.get_body_state()
    round_log       = []
    convergence_log = []

    for r in range(cfg.rounds):
        round_log.append({
            "seed": seed, "round": r + 1, "body_frozen": True, "n_updates": 0,
        })
        tmp = new_model(use_pi)
        tmp.set_body_state(frozen_body)
        convergence_log.extend(
            eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

    models = [new_model(use_pi) for _ in splits]
    for i, (m, cd) in enumerate(zip(models, splits)):
        m.set_body_state(frozen_body)
        m.reset_head(0.0)
        recalib_head(m, cd, std, cfg, seed + 8000 + i, use_pi)

    gm = eval_global(
        global_eval_model(frozen_body, splits, std, cfg, seed, use_pi),
        std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe,
    )
    client_flat, per_client = eval_per_client_personalised(
        models, splits, std, I_probe, x_probe)
    per_client_log = [
        {"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_updates"]  = 0
    metrics["n_rounds"]   = cfg.rounds
    heads = get_heads(models, body_state=frozen_body)

    return metrics, round_log, [], [], per_client_log, convergence_log, None, frozen_body, heads


# ── Vanilla Sequential ────────────────────────────────────────────────────────

def run_vanilla_sequential(
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
    use_pi = False
    n      = len(splits)
    models = [new_model(use_pi) for _ in range(n)]

    if cfg.pretrain_ep > 0:
        C.train_local(
            model=models[0], X_norm=splits[0].X_train,
            y_norm=std.transform_y(splits[0].y_train), I_phys=splits[0].I_train,
            n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed, use_mono_loss=False, beta_mono=0.0,
            use_hard_guard=use_pi, device=DEVICE, std=std,
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
        for i, (m, cd) in enumerate(zip(models, splits)):
            C.set_seed(seed + r * 100 + i + 1)
            m.set_body_state(cyc_body)
            C.train_local(
                model=m, X_norm=cd.X_train,
                y_norm=std.transform_y(cd.y_train), I_phys=cd.I_train,
                n_epochs=cfg.local_epochs, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                seed=seed + r * 100 + i + 1,
                use_mono_loss=False, beta_mono=0.0,
                use_hard_guard=use_pi, device=DEVICE, std=std,
            )
            cyc_body = m.get_body_state()

        round_log.append({"seed": seed, "round": r + 1})
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
        {"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds
    heads = get_heads(models, body_state=final_body)

    return metrics, round_log, [], [], per_client_log, convergence_log, None, final_body, heads


# ── PI-Vanilla Sequential (corrected) ─────────────────────────────────────────

def run_pi_vanilla_sequential(
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
    n      = len(splits)
    models = [new_model(use_pi) for _ in range(n)]

    if cfg.pretrain_ep > 0:
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
        {"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]

    metrics = {**gm, **client_flat}
    metrics["n_accepted"] = None
    metrics["n_rounds"]   = cfg.rounds
    heads = get_heads(models, body_state=final_body)

    return metrics, round_log, [], [], per_client_log, convergence_log, None, final_body, heads


# ── PI-SDFL Full ──────────────────────────────────────────────────────────────

def run_pi_sdfl_full(
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
    n      = len(splits)
    models = [new_model(cfg.use_hard_guard) for _ in range(n)]

    if cfg.pretrain_ep > 0:
        C.train_local(
            model=models[0], X_norm=splits[0].X_train,
            y_norm=std.transform_y(splits[0].y_train), I_phys=splits[0].I_train,
            n_epochs=cfg.pretrain_ep, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
            seed=seed,
            use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
            use_hard_guard=cfg.use_hard_guard, device=DEVICE, std=std,
            I_probe_phys=(I_probe if cfg.use_mono_loss else None),
            x_probe_norm=(x_probe if cfg.use_mono_loss else None),
        )
    for i in range(1, n):
        models[i].reset_head(0.0)

    ref_body  = models[0].get_body_state()
    best_body = C.clone_state(ref_body)
    cyc_body  = C.clone_state(ref_body)

    init_scores = []
    for i, cd in enumerate(splits):
        sc = C.score_candidate(
            body_state=ref_body, head_init_val=0.0,
            X_calib_norm=cd.X_calib, y_calib_raw=cd.y_calib, I_calib_phys=cd.I_calib,
            X_audit_norm=cd.X_audit, y_audit_raw=cd.y_audit, I_audit_phys=cd.I_audit,
            I_probe_phys=I_probe, x_probe_norm=x_probe,
            std=std, device=DEVICE, seed=seed + 99000 + i,
            calib_epochs=cfg.calib_epochs, lr_head=cfg.lr_head,
            use_hard_guard=cfg.use_hard_guard,
        )
        init_scores.append(sc)
    r2_best       = float(np.mean([s["R2"] for s in init_scores]))
    n_best_updates = 0

    round_log       = []
    gate_log        = []
    gate_scores_log = []
    convergence_log = []
    prev_snap       = [None] * n

    for r in range(cfg.rounds):
        mu_r = mu_schedule(r, cfg)

        for i in range(n):
            C.set_seed(seed + r * 100 + i + 1)
            models[i].set_body_state(cyc_body)
            anchor = (C.clone_state(ref_body) if mu_r > 0.0 else None)

            C.train_local(
                model=models[i], X_norm=splits[i].X_train,
                y_norm=std.transform_y(splits[i].y_train), I_phys=splits[i].I_train,
                n_epochs=cfg.local_epochs, lr_body=cfg.lr_body, lr_head=cfg.lr_head,
                seed=seed + r * 100 + i + 1,
                use_mono_loss=cfg.use_mono_loss, beta_mono=cfg.beta_mono,
                use_hard_guard=cfg.use_hard_guard, device=DEVICE, std=std,
                I_probe_phys=(I_probe if cfg.use_mono_loss else None),
                x_probe_norm=(x_probe if cfg.use_mono_loss else None),
                anchor_body=anchor, mu_anchor=mu_r,
            )
            cyc_body      = models[i].get_body_state()
            prev_snap[i]  = (C.clone_state(cyc_body), float(models[i].alpha_k.detach().item()))

        candidate_body = C.clone_state(cyc_body)
        scores = []
        for i, cd in enumerate(splits):
            sc = C.score_candidate(
                body_state=candidate_body, head_init_val=0.0,
                X_calib_norm=cd.X_calib, y_calib_raw=cd.y_calib, I_calib_phys=cd.I_calib,
                X_audit_norm=cd.X_audit, y_audit_raw=cd.y_audit, I_audit_phys=cd.I_audit,
                I_probe_phys=I_probe, x_probe_norm=x_probe,
                std=std, device=DEVICE, seed=seed + r * 100 + 50 + i,
                calib_epochs=cfg.calib_epochs, lr_head=cfg.lr_head,
                use_hard_guard=cfg.use_hard_guard,
            )
            scores.append(sc)
            gate_scores_log.append({
                "seed":                seed,
                "round":               r + 1,
                "client":              i,
                "R2":                  sc["R2"],
                "mono_viol_rate":      sc["mono_viol_rate"],
                "mono_viol_magnitude": sc["mono_viol_magnitude"],
            })

        macro_r2 = float(np.mean([s["R2"] for s in scores]))
        max_pvr  = float(np.max([s["mono_viol_rate"] for s in scores]))
        max_mag  = float(np.max([s["mono_viol_magnitude"] for s in scores]))

        r2_best_before = r2_best
        gate = C.trust_gate_decision(
            {"R2_macro": macro_r2,
             "mono_viol_rate_worst": max_pvr,
             "mono_viol_magnitude_worst": max_mag},
            r2_best=r2_best,
            tau_pvr=cfg.tau_pvr, tau_mag=cfg.tau_mag,
            epsilon_r2=cfg.epsilon_r2,
        )

        if gate["accept"]:
            ref_body = C.clone_state(candidate_body)
            if macro_r2 > r2_best:
                r2_best        = macro_r2
                best_body      = C.clone_state(ref_body)
                n_best_updates += 1
        else:
            prev_snap = [None] * n

        cyc_body = C.clone_state(ref_body)

        gate_log.append({
            "seed":             seed,
            "round":            r + 1,
            "macro_r2":         macro_r2,
            "max_pvr":          max_pvr,
            "max_mag":          max_mag,
            "r2_best_before":   r2_best_before,
            "r2_best_after":    r2_best,
            "rollback_applied": not gate["accept"],
            **gate,
        })
        round_log.append({
            "seed":     seed,
            "round":    r + 1,
            "mu_anchor": mu_r,
            "accepted": bool(gate["accept"]),
            "r2_best":  r2_best,
        })

        tmp = new_model(cfg.use_hard_guard)
        tmp.set_body_state(ref_body)
        convergence_log.extend(
            eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

    final_models = [new_model(cfg.use_hard_guard) for _ in range(n)]
    for i, (fm, cd) in enumerate(zip(final_models, splits)):
        fm.set_body_state(best_body)
        fm.reset_head(0.0)
        recalib_head(fm, cd, std, cfg, seed + 9000 + i, cfg.use_hard_guard)

    gm = eval_global(
        global_eval_model(best_body, splits, std, cfg, seed, cfg.use_hard_guard),
        std, X_eval_norm, y_eval_raw, I_eval_phys, I_probe, x_probe,
    )
    client_flat, per_client = eval_per_client_personalised(
        final_models, splits, std, I_probe, x_probe)
    per_client_log = [
        {"seed": seed, "client": i, **pm} for i, pm in enumerate(per_client)]

    n_accepted = sum(1 for g in gate_log if g.get("accept") is True)

    metrics = {**gm, **client_flat}
    metrics["r2_best_gate"] = float(r2_best)
    metrics["n_accepted"]    = n_accepted
    metrics["n_best_updates"] = n_best_updates
    metrics["n_rounds"]     = cfg.rounds
    heads = get_heads(final_models, body_state=best_body)

    return (metrics, round_log, gate_log, [], per_client_log,
            gate_scores_log, convergence_log, best_body, ref_body, heads)


# ── Per-severity runner ───────────────────────────────────────────────────────

RUNNER_MAP = {
    "Frozen-Reference":       run_frozen_reference,
    "Vanilla-Sequential":     run_vanilla_sequential,
    "PI-Vanilla-Sequential":  run_pi_vanilla_sequential,
    "PI-SDFL-full":           run_pi_sdfl_full,
}


def run_one_severity(
    frac:     float,
    cfg:      ExperimentConfig,
    out_root: str,
    eval_file: str,
) -> None:
    frac_tag = f"frac_{int(round(frac * 100)):03d}"
    out_dir  = os.path.join(out_root, frac_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SEVERITY  frac={frac:.2f}  q={FIXED_QUANTILE:.2f}  tag={frac_tag}")
    print(f"  mode={cfg.mode}  seeds={list(cfg.seeds)}  rounds={cfg.rounds}")
    print(f"{'='*60}")

    agg = {name: [] for name in RUNNER_MAP}

    for seed in cfg.seeds:
        print(f"\n  SEED {seed}")

        splits_clean, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(
            cfg, eval_file=eval_file, split_seed=0)
        manifest_clean = build_split_manifest(cfg, splits_clean, eval_file)

        if frac > 0.0:
            splits, diag = CU.apply_corruption(
                splits_clean, frac=frac, quantile=FIXED_QUANTILE)
            CU.print_corruption_summary(diag)
            manifest = {**manifest_clean, "corruption": diag}
        else:
            splits   = splits_clean
            manifest = {**manifest_clean,
                        "corruption": {"severity": 0.0, "corruption_type": "none"}}

        I_probe, x_probe = C.build_probe_grid(std, DEVICE)

        for name, runner in RUNNER_MAP.items():
            print(f"    [{name}]...", end=" ", flush=True)
            C.set_seed(seed)

            if name == "PI-SDFL-full":
                result = runner(splits, std, cfg, seed,
                                X_eval_norm, y_eval_raw, I_eval_phys,
                                I_probe, x_probe)
                m, rl, gl, pfl, pcl, gsl, cvl, bb, fb, hd = result
                save_run(
                    out_dir, name, seed, cfg, m, rl,
                    gate_log=gl, pf_log=pfl,
                    per_client_log=pcl, convergence_log=cvl,
                    best_body=bb, final_body=fb,
                    personalized_heads=hd, std=std, split_manifest=manifest,
                )
            else:
                result = runner(splits, std, cfg, seed,
                                X_eval_norm, y_eval_raw, I_eval_phys,
                                I_probe, x_probe)
                m, rl, gl, pfl, pcl, cvl, bb, fb, hd = result
                save_run(
                    out_dir, name, seed, cfg, m, rl,
                    gate_log=gl, pf_log=pfl,
                    per_client_log=pcl, convergence_log=cvl,
                    best_body=bb, final_body=fb,
                    personalized_heads=hd, std=std, split_manifest=manifest,
                )

            agg[name].append(m)
            print_result_line(name, m, cfg, gate=(name == "PI-SDFL-full"))

    aggregate_and_save_summary(agg, out_dir)
    print(f"\n  Saved -> {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(mode: str, fracs: List[float]) -> None:
    cfg      = make_config(mode)
    out_root = os.path.join(RESULTS_DIR, "severity_sweep")
    os.makedirs(out_root, exist_ok=True)

    eval_file = "global_test.csv" if mode == "final" else "global_val.csv"

    print(f"\nSeverity sweep | mode={mode} | fracs={fracs} | q={FIXED_QUANTILE}")

    for frac in fracs:
        run_one_severity(frac, cfg, out_root, eval_file)

    print(f"\nSeverity sweep complete. Results -> {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fault severity sweep for PI-SDFL AEI experiments")
    parser.add_argument(
        "--mode", type=str, default="pilot",
        choices=["smoke", "pilot", "final"],
    )
    parser.add_argument(
        "--fracs", nargs="+", type=float, default=DEFAULT_FRACS,
        help="Fault severity levels to sweep (frac in [0,1]). Default: 0.00 0.25 0.35 0.50 0.75 1.00",
    )
    args = parser.parse_args()

    fracs_validated = []
    for f in args.fracs:
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"--fracs values must be in [0, 1], got {f}")
        fracs_validated.append(f)

    main(args.mode, sorted(set(fracs_validated)))
