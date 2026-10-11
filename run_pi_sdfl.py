"""
run_pi_sdfl.py — PI-SDFL proposed method and ablations.

Runs six variants in one script:
    PI-SDFL-full            All components active (proposed method)
    PI-SDFL-no-anchor       Anchor regularisation disabled
    PI-SDFL-no-gate         Trust gate disabled
    PI-SDFL-no-mono         Monotonicity loss (L_mono) disabled
    PI-SDFL-no-warmstart    Pretrain skipped (random initialisation)
    PI-SDFL-no-gate-no-mono Gate and L_mono both disabled

Outputs per seed per variant:
    final_metrics.json      All scalar metrics including per-client R2/RMSE/PVR
    round_metrics.csv       Per-round gate decision and r2_best
    gate_log.csv            Per-round gate scores and accept/reject
    gate_client_scores.csv  Per-round per-client candidate scores
    forgetting_log.csv      Per-step physics forgetting
    activity_log.csv        Per-(round,client) losses and PVR before/after
    per_client_metrics.csv  Final per-client R2, RMSE, PVR, neg_rate
    convergence_log.csv     Per-client R2 and PVR tracked across rounds
    best_body.pt            Deployed body (highest-scoring accepted)
    final_body.pt           Last trusted reference body

Usage:
    python run_pi_sdfl.py --mode smoke
    python run_pi_sdfl.py --mode pilot
    python run_pi_sdfl.py --mode final
    python run_pi_sdfl.py --mode smoke --variants full no-gate
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
from experiment_utils import (
    ExperimentConfig, make_config, load_data, build_split_manifest,
    ClientSplit, DEVICE, RESULTS_DIR,
    new_model, mu_schedule, recalib_head, get_heads, global_eval_model,
    eval_global, eval_per_client_personalised, eval_per_client_at_round,
    save_run, aggregate_and_save_summary, print_result_line,
)


ALL_VARIANTS = [
    "full",
    "no-anchor",
    "no-gate",
    "no-mono",
    "no-warmstart",
    "no-gate-no-mono",
]

METHOD_NAMES = {
    "full":            "PI-SDFL-full",
    "no-anchor":       "PI-SDFL-no-anchor",
    "no-gate":         "PI-SDFL-no-gate",
    "no-mono":         "PI-SDFL-no-mono",
    "no-warmstart":    "PI-SDFL-no-warmstart",
    "no-gate-no-mono": "PI-SDFL-no-gate-no-mono",
}


def make_variant_config(base: ExperimentConfig, variant: str) -> ExperimentConfig:
    if variant == "full":
        return base
    if variant == "no-anchor":
        return replace(base, use_anchor=False,
                       mu_anchor_start=0.0, mu_anchor_floor=0.0)
    if variant == "no-gate":
        return replace(base, use_gate=False)
    if variant == "no-mono":
        return replace(base, use_mono_loss=False, beta_mono=0.0)
    if variant == "no-warmstart":
        return replace(base, pretrain_ep=0)
    if variant == "no-gate-no-mono":
        return replace(base, use_gate=False, use_mono_loss=False, beta_mono=0.0)
    raise ValueError(f"Unknown variant '{variant}'")


def run_single(
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
        {"seed": seed, "client": i, **{k: v for k, v in pm.items()}}
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


def main(mode: str, variants: List[str]) -> None:
    base_cfg = make_config(mode)
    run_name = mode
    out_dir  = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    eval_file = "global_test.csv" if mode == "final" else "global_val.csv"

    print(f"\n{'='*60}")
    print(f"  PI-SDFL  |  mode={mode}  |  device={DEVICE}")
    print(f"  seeds={list(base_cfg.seeds)}  rounds={base_cfg.rounds}")
    print(f"  variants={variants}")
    print(f"{'='*60}\n")

    agg = {METHOD_NAMES[v]: [] for v in variants}

    for seed in base_cfg.seeds:
        print(f"\n  SEED {seed}")
        splits, std, X_eval_norm, y_eval_raw, I_eval_phys = load_data(
            base_cfg, eval_file=eval_file, split_seed=0)
        manifest = build_split_manifest(base_cfg, splits, eval_file)

        I_probe, x_probe = C.build_probe_grid(std, DEVICE)

        for variant in variants:
            method  = METHOD_NAMES[variant]
            cfg     = make_variant_config(base_cfg, variant)
            verbose = (variant == "full" and seed == base_cfg.seeds[0])

            print(f"  [{method}]...", end=" ", flush=True)
            C.set_seed(seed)
            result = run_single(
                splits=splits, std=std, cfg=cfg, seed=seed,
                X_eval_norm=X_eval_norm, y_eval_raw=y_eval_raw,
                I_eval_phys=I_eval_phys,
                I_probe=I_probe, x_probe=x_probe,
                verbose=verbose,
            )
            (m, rl, gl, pfl, al, gsl, cvl, pcl, bb, fb, hd) = result
            agg[method].append(m)

            save_run(
                out_dir, method, seed, cfg, m, rl,
                gate_log=gl, pf_log=pfl,
                activity_log=al, gate_scores_log=gsl,
                per_client_log=pcl, convergence_log=cvl,
                best_body=bb, final_body=fb,
                personalized_heads=hd, std=std, split_manifest=manifest,
            )
            print_result_line(method, m, cfg, gate=cfg.use_gate)

        print(f"  SEED {seed} COMPLETE")

    aggregate_and_save_summary(agg, out_dir)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PI-SDFL proposed method and ablations")
    parser.add_argument(
        "--mode", type=str, default="smoke",
        choices=["smoke", "pilot", "final"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help="Which variants to run. Default: all.",
    )
    args = parser.parse_args()
    main(args.mode, args.variants)