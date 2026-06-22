#!/usr/bin/env python3
"""GATE-TRAJ: Can we predict future weight states in aligned weight space?

Tests the prerequisite for the entire aligned-weight-space program:
if trajectory prediction doesn't work in aligned space, neither JEPA
(trajectory or scale) has a foundation.

Flow:
  1. Load trajectories (N, T, D) — checkpoints along training
  2. Align: one weight-match per trajectory (last ckpt → reference), apply to all
  3. Train/test split (80/20 by trajectory)
  4. Build training pairs (w_t, w_{t+n}) from train trajectories
  5. Fit predictors:
     a) identity: predict w_{t+n} = w_t (baseline — should be worst)
     b) avg-velocity aligned: predicted = w_t_aligned + mean_delta_aligned
     c) avg-velocity raw: predicted = w_t + mean_delta_raw
     d) pca-ridge aligned: PCA(k) + ridge regression in aligned space
     e) pca-ridge raw: same in raw space
  6. On test trajectories: predict w_{t+n}, decode, eval on task
  7. Report: predicted CE vs actual-t CE, aligned vs raw

GATE PASSES if:
  - any predictor's CE < actual-t CE (predicted future beats current)
  - aligned predictor CE < raw predictor CE (alignment helps)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.backends.mps

from src.align.rebasin import (
    build_layout,
    canonical_sort_perms,
    weight_match_perms,
    apply_perms_to_flat,
)
from src.arch_gen.spec import ArchSpec  # noqa: F401 — forces package init
from src.models.variable_mlp import VariableTinyMLP
from data.synthetic_text import VOCAB_SIZE
from src.utils.weights import load_flat_into_model
from src.utils.train_loop import eval_model_loss, set_seed


def make_model_factory(hidden: int, depth: int):
    spec = ArchSpec(hidden, depth, skip=False)
    return lambda: VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)


def align_trajectories(trajs: torch.Tensor, meta: dict, H: int, depth: int) -> torch.Tensor:
    """Align all trajectories to canonical reference. One perm per trajectory.

    Reference = canonical_sort of traj_0, ckpt_0.
    For each trajectory: weight-match LAST checkpoint to reference, apply that
    perm to ALL checkpoints (neurons don't permute during training).
    """
    N, T, D = trajs.shape
    layout = build_layout(meta, H, depth)

    # Canonicalize reference
    ref_perms = canonical_sort_perms(trajs[0, 0], layout)
    ref = apply_perms_to_flat(trajs[0, 0], ref_perms, layout)

    aligned = torch.empty_like(trajs)
    for i in range(N):
        # Match last checkpoint to reference (most trained = best match)
        perm = weight_match_perms(trajs[i, -1], ref, layout, iters=3)
        # Apply same perm to all checkpoints in this trajectory
        for t in range(T):
            aligned[i, t] = apply_perms_to_flat(trajs[i, t], perm, layout)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  aligned {i+1}/{N} trajectories")

    return aligned


def eval_weights(flat: torch.Tensor, factory, device: str, seed: int) -> float:
    """Load flat weights into a fresh model, eval on task, return CE loss."""
    m = factory().to(device)
    load_flat_into_model(flat, m)
    result = eval_model_loss(m, seed=seed, device=device)
    return result["eval_loss"]


def run_traj_gate(
    trajs: torch.Tensor,
    meta: dict,
    *,
    hidden: int,
    depth: int,
    predict_horizon: int = 200,
    pca_dim: int = 64,
    ridge_alpha: float = 10.0,
    n_train: int = 80,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    N, T, D = trajs.shape
    H = hidden
    ckpt_interval = meta["ckpt_interval"]
    n_ckpts_ahead = predict_horizon // ckpt_interval

    print(f"Trajectories: {tuple(trajs.shape)} | predict {n_ckpts_ahead} ckpts ahead ({predict_horizon} steps)")
    print(f"PCA dim={pca_dim} ridge_alpha={ridge_alpha} train={n_train} test={N - n_train}")

    # Train/test split by trajectory
    set_seed(seed)
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # --- Align ---
    print("Aligning trajectories...")
    aligned = align_trajectories(trajs, meta, H, depth)

    factory = make_model_factory(H, depth)

    # --- Build training pairs (w_t, w_{t+n}) ---
    # t ranges from 0 to T-1-n_ckpts_ahead
    max_t = T - n_ckpts_ahead - 1
    print(f"Training pairs: t in [0, {max_t}], target = t + {n_ckpts_ahead}")

    # Aligned pairs
    train_X_aln = []
    train_Y_aln = []
    for i in train_idx:
        for t in range(max_t + 1):
            train_X_aln.append(aligned[i, t])
            train_Y_aln.append(aligned[i, t + n_ckpts_ahead])
    train_X_aln = torch.stack(train_X_aln)
    train_Y_aln = torch.stack(train_Y_aln)

    # Raw pairs
    train_X_raw = []
    train_Y_raw = []
    for i in train_idx:
        for t in range(max_t + 1):
            train_X_raw.append(trajs[i, t])
            train_Y_raw.append(trajs[i, t + n_ckpts_ahead])
    train_X_raw = torch.stack(train_X_raw)
    train_Y_raw = torch.stack(train_Y_raw)

    # --- Predictors ---

    # (a) avg velocity aligned
    delta_aln = (train_Y_aln - train_X_aln).mean(dim=0)
    # (b) avg velocity raw
    delta_raw = (train_Y_raw - train_X_raw).mean(dim=0)

    # (c) PCA + ridge aligned
    all_aln = aligned.reshape(-1, D)
    mean_aln = all_aln.mean(dim=0)
    centered_aln = all_aln - mean_aln
    U_aln, S_aln, Vh_aln = torch.linalg.svd(centered_aln, full_matrices=False)
    comps_aln = Vh_aln[:pca_dim]  # (k, D)
    pc_X_aln = (train_X_aln - mean_aln) @ comps_aln.t()
    pc_Y_aln = (train_Y_aln - mean_aln) @ comps_aln.t()
    W_aln, b_aln = _ridge(pc_X_aln, pc_Y_aln, ridge_alpha)

    # (d) PCA + ridge raw
    all_raw = trajs.reshape(-1, D)
    mean_raw = all_raw.mean(dim=0)
    centered_raw = all_raw - mean_raw
    U_raw, S_raw, Vh_raw = torch.linalg.svd(centered_raw, full_matrices=False)
    comps_raw = Vh_raw[:pca_dim]
    pc_X_raw = (train_X_raw - mean_raw) @ comps_raw.t()
    pc_Y_raw = (train_Y_raw - mean_raw) @ comps_raw.t()
    W_raw, b_raw = _ridge(pc_X_raw, pc_Y_raw, ridge_alpha)

    # --- Evaluate on test trajectories ---
    arms = ["identity", "avg_vel_aln", "avg_vel_raw", "pca_ridge_aln", "pca_ridge_raw"]
    results = {arm: [] for arm in arms}

    print(f"\nEvaluating {len(test_idx)} test trajectories × {max_t + 1} t-values...")
    for idx_i, i in enumerate(test_idx):
        for t in range(max_t + 1):
            w_t = trajs[i, t]
            w_t_aln = aligned[i, t]
            eval_seed = seed + int(i) * 100 + t

            # (a) identity — use current weights (baseline)
            results["identity"].append(eval_weights(w_t, factory, device, eval_seed))

            # (b) avg velocity aligned
            pred_aln = w_t_aln + delta_aln
            results["avg_vel_aln"].append(eval_weights(pred_aln, factory, device, eval_seed))

            # (c) avg velocity raw
            pred_raw = w_t + delta_raw
            results["avg_vel_raw"].append(eval_weights(pred_raw, factory, device, eval_seed))

            # (d) PCA + ridge aligned
            pc_t = (w_t_aln - mean_aln) @ comps_aln.t()
            pc_pred = pc_t @ W_aln + b_aln
            pred_pca_aln = (pc_pred @ comps_aln) + mean_aln
            results["pca_ridge_aln"].append(eval_weights(pred_pca_aln.reshape(-1)[:D], factory, device, eval_seed))

            # (e) PCA + ridge raw
            pc_t_raw = (w_t - mean_raw) @ comps_raw.t()
            pc_pred_raw = pc_t_raw @ W_raw + b_raw
            pred_pca_raw = (pc_pred_raw @ comps_raw) + mean_raw
            results["pca_ridge_raw"].append(eval_weights(pred_pca_raw.reshape(-1)[:D], factory, device, eval_seed))

        if (idx_i + 1) % 5 == 0 or idx_i == 0:
            print(f"  test traj {idx_i+1}/{len(test_idx)} done")

    # --- Summary ---
    summary = {
        "N": N, "T": T, "D": D,
        "hidden": H, "depth": depth,
        "predict_horizon": predict_horizon,
        "n_ckpts_ahead": n_ckpts_ahead,
        "pca_dim": pca_dim,
        "ridge_alpha": ridge_alpha,
        "n_train": n_train,
        "n_test": len(test_idx),
        "n_eval_points": len(results["identity"]),
    }

    for arm in arms:
        vals = results[arm]
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[arm] = {"ce_mean": m, "ce_std": sd, "n": len(vals)}

    # Deltas vs identity (positive = predictor beats identity = predicted future better than current)
    id_mean = summary["identity"]["ce_mean"]
    for arm in arms:
        if arm == "identity":
            continue
        arm_mean = summary[arm]["ce_mean"]
        summary[arm]["delta_vs_identity"] = id_mean - arm_mean  # positive = better than identity

    # Aligned vs raw comparison
    summary["alignment_helps_avg_vel"] = summary["avg_vel_aln"]["ce_mean"] < summary["avg_vel_raw"]["ce_mean"]
    summary["alignment_helps_pca_ridge"] = summary["pca_ridge_aln"]["ce_mean"] < summary["pca_ridge_raw"]["ce_mean"]

    # Gate verdict
    best_predictor = min(arms, key=lambda a: summary[a]["ce_mean"])
    best_delta = id_mean - summary[best_predictor]["ce_mean"]
    summary["best_predictor"] = best_predictor
    summary["best_delta_vs_identity"] = best_delta
    summary["gate_passes"] = best_delta > 0.01  # predicted future beats current by >0.01 CE
    summary["alignment_helps"] = (
        summary["avg_vel_aln"]["ce_mean"] < summary["avg_vel_raw"]["ce_mean"]
        and summary["pca_ridge_aln"]["ce_mean"] < summary["pca_ridge_raw"]["ce_mean"]
    )

    return summary


def _ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression: Y ≈ X @ W + b. Returns (W, b).

    X: (M, k_in), Y: (M, k_out)
    """
    # Augment with bias column
    M, k_in = X.shape
    k_out = Y.shape[1]
    X_aug = torch.cat([X, torch.ones(M, 1, dtype=X.dtype)], dim=1)  # (M, k_in+1)
    reg = alpha * torch.eye(k_in + 1, dtype=X.dtype)
    reg[-1, -1] = 0.0  # don't regularize bias
    W_aug = torch.linalg.solve(X_aug.t() @ X_aug + reg, X_aug.t() @ Y)
    return W_aug[:k_in], W_aug[k_in:]  # (k_in, k_out), (k_out,)


def main():
    p = argparse.ArgumentParser(description="GATE-TRAJ: trajectory prediction in aligned weight space")
    p.add_argument("--trajectories", type=str, required=True, help="path to trajectories.pt")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--predict_horizon", type=int, default=200, help="steps ahead to predict")
    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="checkpoints/align/traj_gate")
    args = p.parse_args()

    traj_path = Path(args.trajectories)
    trajs = torch.load(traj_path, weights_only=True)
    meta = json.loads((traj_path.parent / "meta.json").read_text())

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    summary = run_traj_gate(
        trajs, meta,
        hidden=args.hidden, depth=args.depth,
        predict_horizon=args.predict_horizon,
        pca_dim=args.pca_dim,
        ridge_alpha=args.ridge_alpha,
        n_train=args.n_train,
        seed=args.seed,
        device=device,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traj_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== GATE-TRAJ SUMMARY ===")
    print(f"Model: depth={summary['depth']} H={summary['hidden']} D={summary['D']}")
    print(f"Predict: {summary['predict_horizon']} steps ahead ({summary['n_ckpts_ahead']} checkpoints)")
    print(f"Train: {summary['n_train']} trajectories | Test: {summary['n_test']} trajectories")
    print(f"Eval points: {summary['n_eval_points']}")
    print()
    for arm in ["identity", "avg_vel_aln", "avg_vel_raw", "pca_ridge_aln", "pca_ridge_raw"]:
        s = summary[arm]
        delta = s.get("delta_vs_identity", 0.0)
        d_str = f"  delta_vs_id={delta:+.4f}" if arm != "identity" else ""
        print(f"  {arm:20s}  CE={s['ce_mean']:.4f} ± {s['ce_std']:.4f}{d_str}")
    print()
    print(f"Best predictor: {summary['best_predictor']}  delta={summary['best_delta_vs_identity']:+.4f}")
    print(f"GATE PASSES (best > identity by >0.01): {summary['gate_passes']}")
    print(f"Alignment helps (both arms): {summary['alignment_helps']}")
    print(f"Artifacts: {out_dir}/traj_gate_summary.json")


if __name__ == "__main__":
    main()
