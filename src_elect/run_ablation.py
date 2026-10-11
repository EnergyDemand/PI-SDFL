"""
run_ablation.py — PI-SDFL ablation study for AEI Table 7.

Runs five ablation variants of PI-SDFL on the synthetic PEM dataset
under the same AEI frozen protocol as the clean comparison (10 seeds,
20 rounds). PI-SDFL-full results are NOT rerun here — they already
exist in results_new/final/PI-SDFL-full/ from the clean comparison run.

Ablation variants (one change from full PI-SDFL per row):
    PI-SDFL-no-anchor      : use_anchor=False  (no anchor regularisation)
    PI-SDFL-no-gate        : use_gate=False    (no trust gate, no rollback)
    PI-SDFL-no-mono        : use_mono_loss=False, beta_mono=0 (no L_mono)
    PI-SDFL-no-warmstart   : pretrain_ep=0     (no Client-1 warm-start)
    PI-SDFL-no-gate-no-mono: use_gate=False + use_mono_loss=False (no trust gate and no
                              monotonicity loss; hard guard retained)

All variants use the PI architecture (use_hard_guard=True).
Each variant differs from PI-SDFL-full in EXACTLY ONE design choice,
verified by assert_exact_change() at startup.

Results are saved under results_new/final/<variant-name>/seed_XXX/
so they sit alongside existing method folders and are picked up by
aggregate_clean_results.py (which includes all 8 method names).

Usage:
    python run_ablation.py --mode smoke
    python run_ablation.py --mode pilot
    python run_ablation.py --mode final

Place this file in:
    D:\\GPU\\pi_sdfl_v2\\src_elect_new\\run_ablation.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from experiment_utils import (
    ExperimentConfig, make_config, load_data, build_split_manifest,
    ClientSplit, DEVICE, RESULTS_DIR,
    new_model, mu_schedule, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, print_result_line,
)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

def make_ablation_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
    """
    Build the five ablation configs from the base (full PI-SDFL) config.
    Each variant differs from base in EXACTLY the listed fields.
    assert_exact_change() verifies this at runtime.
    """
    variants = {
        "PI-SDFL-no-anchor": replace(
            base,
            use_anchor=False,
            mu_anchor_start=0.0,
            mu_anchor_floor=0.0,
        ),
        "PI-SDFL-no-gate": replace(
            base,
            use_gate=False,
        ),
        "PI-SDFL-no-mono": replace(
            base,
            use_mono_loss=False,
            beta_mono=0.0,
        ),
        "PI-SDFL-no-warmstart": replace(
            base,
            pretrain_ep=0,
        ),
        "PI-SDFL-no-gate-no-mono": replace(
            base,
            use_gate=False,
            use_mono_loss=False,
            beta_mono=0.0,
        ),
    }

    expected_changes = {
        "PI-SDFL-no-anchor":       ["use_anchor", "mu_anchor_start", "mu_anchor_floor"],
        "PI-SDFL-no-gate":         ["use_gate"],
        "PI-SDFL-no-mono":         ["use_mono_loss", "beta_mono"],
        "PI-SDFL-no-warmstart":    ["pretrain_ep"],
        "PI-SDFL-no-gate-no-mono": ["use_gate", "use_mono_loss", "beta_mono"],
    }

    for name, cfg_abl in variants.items():
        actual = {
            f.name for f in fields(base)
            if getattr(base, f.name) != getattr(cfg_abl, f.name)
        }
        expected = set(expected_changes[name])
        if actual != expected:
            raise AssertionError(
                f"Ablation '{name}' config error:\n"
                f"  Expected changes : {sorted(expected)}\n"
                f"  Actual changes   : {sorted(actual)}\n"
                f"  Extra            : {sorted(actual - expected)}\n"
                f"  Missing          : {sorted(expected - actual)}"
            )

    return variants


# ---------------------------------------------------------------------------
# PI-SDFL ring runner (identical to run_pi_sdfl.py run_single)
# ---------------------------------------------------------------------------

def run_pi_sdfl_variant(
    splits:      List[ClientSplit],
    std:         C.Standardizer,
    cfg:         ExperimentConfig,
    seed:        int,
    X_eval_norm: np.ndarray,
    y_eval_raw:  np.ndarray,
    I_eval_phys: np.ndarray,
    I_probe:     torch.Tensor,
    x_probe:     torch.Tensor,
    verbose:     bool = False,
) -> Tuple:
    n = len(splits)
    C.set_seed(seed)
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

    if cfg.use_gate:
        init_scores = []
        for i, cd in enumerate(splits):
            sc = C.score_candidate(
                body_state=ref_body, head_init_val=0.0,
                X_calib_norm=cd.X_calib, y_calib_raw=cd.y_calib,
                I_calib_phys=cd.I_calib,
                X_audit_norm=cd.X_audit, y_audit_raw=cd.y_audit,
                I_audit_phys=cd.I_audit,
                I_probe_phys=I_probe, x_probe_norm=x_probe,
                std=std, device=DEVICE,
                seed=seed + 99000 + i,
                calib_epochs=cfg.calib_epochs, lr_head=cfg.lr_head,
                use_hard_guard=cfg.use_hard_guard,
            )
            init_scores.append(sc)
        r2_best = float(np.mean([s["R2"] for s in init_scores]))
    else:
        r2_best = -np.inf

    round_log:       List[Dict] = []
    gate_log:        List[Dict] = []
    gate_scores_log: List[Dict] = []
    pf_log:          List[Dict] = []
    activity_log:    List[Dict] = []
    convergence_log: List[Dict] = []

    global_step = 0
    prev_snap: List[Optional[Tuple]] = [None] * n

    for r in range(cfg.rounds):
        mu_r = mu_schedule(r, cfg)

        for i in range(n):
            C.set_seed(seed + r * 100 + i + 1)
            body_incoming = C.clone_state(cyc_body)
            models[i].set_body_state(cyc_body)
            anchor = (C.clone_state(ref_body) if mu_r > 0.0 else None)

            head_before = float(models[i].alpha_k.detach().item())
            pm_before   = C.physics_metrics(models[i], I_probe, x_probe, std, DEVICE)

            loss_dict = C.train_local(
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

            body_after = models[i].get_body_state()
            head_after = float(models[i].alpha_k.detach().item())

            with torch.no_grad():
                models[i].alpha_k.fill_(head_before)
            pm_after = C.physics_metrics(models[i], I_probe, x_probe, std, DEVICE)
            with torch.no_grad():
                models[i].alpha_k.fill_(head_after)

            activity_log.append({
                "seed":              seed,
                "round":             r + 1,
                "client":            i,
                "l_data":            loss_dict["l_data"],
                "l_mono":            loss_dict["l_mono"],
                "anchor_weighted":   mu_r * loss_dict["l_anchor"],
                "pvr_before":        pm_before["mono_viol_rate"],
                "pvr_after":         pm_after["mono_viol_rate"],
                "mag_before":        pm_before["mono_viol_magnitude"],
                "mag_after":         pm_after["mono_viol_magnitude"],
                "neg_rate_before":   pm_before["neg_rate"],
                "neg_rate_after":    pm_after["neg_rate"],
                "body_hash":         C.state_hash(body_after),
                "round_gate_accept": None,
            })

            if prev_snap[i] is not None:
                body_prev, head_prev = prev_snap[i]
                pf = C.body_forgetting(
                    body_now=body_incoming, body_ref=body_prev,
                    head_ref_val=head_prev,
                    X_audit_norm=splits[i].X_audit,
                    y_audit_raw=splits[i].y_audit,
                    I_audit_phys=splits[i].I_audit,
                    I_probe_phys=I_probe, x_probe_norm=x_probe,
                    std=std, device=DEVICE,
                    use_hard_guard=cfg.use_hard_guard,
                )
                pf_log.append({
                    "seed": seed, "round": r + 1, "client": i,
                    "global_step": global_step, "lag_steps": n - 1,
                    "round_gate_accept": None,
                    **pf,
                })

            prev_snap[i] = (C.clone_state(body_after), head_after)
            cyc_body     = body_after
            global_step += 1

        if cfg.use_gate:
            candidate_body      = C.clone_state(cyc_body)
            candidate_body_hash = C.state_hash(candidate_body)

            scores = []
            for i, cd in enumerate(splits):
                sc = C.score_candidate(
                    body_state=candidate_body, head_init_val=0.0,
                    X_calib_norm=cd.X_calib, y_calib_raw=cd.y_calib,
                    I_calib_phys=cd.I_calib,
                    X_audit_norm=cd.X_audit, y_audit_raw=cd.y_audit,
                    I_audit_phys=cd.I_audit,
                    I_probe_phys=I_probe, x_probe_norm=x_probe,
                    std=std, device=DEVICE,
                    seed=seed + r * 100 + 50 + i,
                    calib_epochs=cfg.calib_epochs, lr_head=cfg.lr_head,
                    use_hard_guard=cfg.use_hard_guard,
                )
                scores.append(sc)
                gate_scores_log.append({
                    "seed":                seed,
                    "round":               r + 1,
                    "client":              i,
                    "R2":                  sc["R2"],
                    "RMSE":                sc.get("RMSE"),
                    "mono_viol_rate":      sc["mono_viol_rate"],
                    "mono_viol_magnitude": sc["mono_viol_magnitude"],
                    "neg_rate":            sc.get("neg_rate"),
                })

            macro_r2 = float(np.mean([s["R2"]                for s in scores]))
            max_pvr  = float(np.max( [s["mono_viol_rate"]    for s in scores]))
            max_mag  = float(np.max( [s["mono_viol_magnitude"] for s in scores]))
            r2_best_before = r2_best

            gate = C.trust_gate_decision(
                {"R2_macro": macro_r2,
                 "mono_viol_rate_worst": max_pvr,
                 "mono_viol_magnitude_worst": max_mag},
                r2_best=r2_best,
                tau_pvr=cfg.tau_pvr, tau_mag=cfg.tau_mag,
                epsilon_r2=cfg.epsilon_r2,
            )
            rollback_applied = not gate["accept"]

            if gate["accept"]:
                ref_body = C.clone_state(candidate_body)
                if macro_r2 > r2_best:
                    r2_best   = macro_r2
                    best_body = C.clone_state(ref_body)
            else:
                prev_snap = [None] * n

            cyc_body = C.clone_state(ref_body)

            gate_log.append({
                "seed":                seed,
                "round":               r + 1,
                "macro_r2":            macro_r2,
                "max_pvr":             max_pvr,
                "max_mag":             max_mag,
                "r2_best_before":      r2_best_before,
                "r2_best_after":       r2_best,
                "accuracy_threshold":  r2_best_before - cfg.epsilon_r2,
                "rollback_applied":    rollback_applied,
                "candidate_body_hash": candidate_body_hash,
                "reference_body_hash": C.state_hash(ref_body),
                **gate,
            })
        else:
            ref_body  = C.clone_state(cyc_body)
            best_body = C.clone_state(ref_body)
            cyc_body  = C.clone_state(ref_body)
            gate      = None
            gate_log.append({
                "seed":             seed,
                "round":            r + 1,
                "accept":           None,
                "reason":           "no_gate",
                "r2_best_before":   r2_best,
                "r2_best_after":    r2_best,
                "rollback_applied": False,
            })

        accept_val = None if gate is None else bool(gate["accept"])
        for row in activity_log:
            if row["round"] == r + 1 and row["round_gate_accept"] is None:
                row["round_gate_accept"] = accept_val
        for row in pf_log:
            if row["round"] == r + 1 and row["round_gate_accept"] is None:
                row["round_gate_accept"] = accept_val

        round_log.append({
            "seed":      seed,
            "round":     r + 1,
            "mu_anchor": mu_r,
            "accepted":  accept_val,
            "r2_best":   r2_best,
        })

        tmp = new_model(cfg.use_hard_guard)
        tmp.set_body_state(ref_body)
        convergence_log.extend(
            eval_per_client_at_round(tmp, splits, std, I_probe, x_probe, seed, r + 1))

        if verbose:
            accept_str = str(accept_val) if accept_val is not None else "no_gate"
            reason_str = "" if gate is None else gate.get("reason", "")
            print(f"    [r{r+1:02d}] mu={mu_r:.2f}  r2_best={r2_best:.4f}  "
                  f"accept={accept_str}  {reason_str}")

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
        {"seed": seed, "client": i, **pm}
        for i, pm in enumerate(per_client)
    ]

    n_accepted = sum(1 for g in gate_log if g.get("accept") is True)
    metrics = {**gm, **client_flat}
    metrics["r2_best_gate"] = float(r2_best) if cfg.use_gate else None
    metrics["n_accepted"]   = n_accepted if cfg.use_gate else None
    metrics["n_rounds"]     = cfg.rounds

    heads = get_heads(final_models, body_state=best_body)

    return (metrics, round_log, gate_log, pf_log, activity_log,
            gate_scores_log, convergence_log, per_client_log,
            best_body, ref_body, heads)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode: str) -> None:
    base_cfg = make_config(mode)
    variants = make_ablation_configs(base_cfg)

    eval_file = "global_test.csv" if mode == "final" else "global_val.csv"
    out_dir   = os.path.join(RESULTS_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  PI-SDFL ABLATION  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(base_cfg.seeds)}  rounds={base_cfg.rounds}")
    print(f"  eval_file={eval_file}")
    print(f"  variants={list(variants.keys())}")
    print(f"{'='*65}\n")

    agg: Dict[str, List[Dict]] = {name: [] for name in variants}

    for seed in base_cfg.seeds:
        print(f"\n{'--'*28}  SEED {seed}")

        splits, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(
            base_cfg, eval_file=eval_file, split_seed=0)
        manifest = build_split_manifest(base_cfg, splits, eval_file)

        I_probe, x_probe = C.build_probe_grid(std, DEVICE)

        for name, cfg_abl in variants.items():
            print(f"  [{name}]...", end=" ", flush=True)
            C.set_seed(seed)

            result = run_pi_sdfl_variant(
                splits=splits, std=std, cfg=cfg_abl, seed=seed,
                X_eval_norm=X_eval_norm, y_eval_raw=y_eval_raw,
                I_eval_phys=I_eval_phys,
                I_probe=I_probe, x_probe=x_probe,
                verbose=False,
            )
            (m, rl, gl, pfl, al, gsl, cvl, pcl, bb, fb, hd) = result
            agg[name].append(m)

            save_run(
                out_dir, name, seed, cfg_abl, m, rl,
                gate_log=gl, pf_log=pfl,
                activity_log=al, gate_scores_log=gsl,
                per_client_log=pcl, convergence_log=cvl,
                best_body=bb, final_body=fb,
                personalized_heads=hd, std=std, split_manifest=manifest,
            )

            r2  = m.get("global_R2")
            pvr = m.get("global_mono_viol_rate")
            acc = m.get("n_accepted")
            r2_s  = "--" if r2  is None else f"{r2:.4f}"
            pvr_s = "--" if pvr is None else f"{pvr:.4f}"
            acc_s = "--" if acc is None else f"{acc}/{cfg_abl.rounds}"
            print(f"R2={r2_s}  PVR={pvr_s}  gate={acc_s}")

        print(f"  SEED {seed} COMPLETE")

    print(f"\n{'='*65}")
    print("ABLATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Variant':<35} {'R²':>10} {'PVR':>8} {'Gate':>8}")
    print(f"  {'-'*63}")
    for name, results in agg.items():
        if not results:
            continue
        r2_vals  = [r["global_R2"]             for r in results if r.get("global_R2") is not None]
        pvr_vals = [r["global_mono_viol_rate"] for r in results if r.get("global_mono_viol_rate") is not None]
        acc_vals = [r["n_accepted"]            for r in results if r.get("n_accepted") is not None]
        r2_s  = f"{np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}" if r2_vals  else "--"
        pvr_s = f"{np.mean(pvr_vals):.4f}"                       if pvr_vals else "--"
        acc_s = f"{np.mean(acc_vals):.1f}/{base_cfg.rounds}"     if acc_vals else "--"
        print(f"  {name:<35} {r2_s:>14} {pvr_s:>8} {acc_s:>8}")

    aggregate_and_save_summary(agg, out_dir)
    print(f"\nAblation results saved -> {out_dir}")
    print("Re-run aggregate_clean_results.py to merge into summary_clean.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PI-SDFL ablation study for AEI Table 7.")
    parser.add_argument(
        "--mode", type=str, default="smoke",
        choices=["smoke", "pilot", "final"],
    )
    args = parser.parse_args()
    main(args.mode)
