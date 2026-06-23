#!/usr/bin/env python3
"""GATE-TRAJ v3: Window-conditioned trajectory prediction.

The v2 result: predicting w_{t+n} from w_t alone fails because trajectories
diverge in nearly orthogonal directions (cosine sim 0.02-0.12). A single
snapshot can't tell where THIS model is going.

Fix: predict from a WINDOW of the last K checkpoints. This gives the predictor
the model's velocity, not just position. If K=1 this reduces to v2. K=3 gives
300 steps of history.

Also fixes the MLP: smaller (128 hidden, 2 layers), heavy regularization
(weight_decay=1e-2, 200 epochs, early stopping on train loss plateau).

Arms: identity, actual_future (ceiling), window_mlp_aln, window_mlp_raw.
Reports headroom, capture%, per-t breakdown.
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


def align_trajectories(trajs, meta, H, depth):
    N, T, D = trajs.shape
    layout = build_layout(meta, H, depth)
    ref_perms = canonical_sort_perms(trajs[0, 0], layout)
    ref = apply_perms_to_flat(trajs[0, 0], ref_perms, layout)
    aligned = torch.empty_like(trajs)
    for i in range(N):
        perm = weight_match_perms(trajs[i, -1], ref, layout, iters=3)
        for t in range(T):
            aligned[i, t] = apply_perms_to_flat(trajs[i, t], perm, layout)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  aligned {i+1}/{N} trajectories")
    return aligned


def eval_weights(flat, factory, device, seed):
    m = factory().to(device)
    load_flat_into_model(flat, m)
    return eval_model_loss(m, seed=seed, device=device)["eval_loss"]


class WindowMLP(torch.nn.Module):
    """Predicts PCA-space w_{t+n} from a window of K PCA-space checkpoints."""
    def __init__(self, window_k, pca_dim, hidden=128, n_layers=2):
        super().__init__()
        layers = [torch.nn.Linear(window_k * pca_dim, hidden), torch.nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.SiLU()]
        layers += [torch.nn.Linear(hidden, pca_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):  # x: (B, K * pca_dim)
        return self.net(x)


def run_window_gate(
    trajs, meta, *, hidden, depth,
    predict_horizon=200, window_k=3, pca_dim=32,
    n_train=80, seed=42, device="cpu",
    mlp_hidden=128, mlp_epochs=300, mlp_wd=1e-2, mlp_lr=1e-3,
):
    N, T, D = trajs.shape
    H = hidden
    ckpt_interval = meta["ckpt_interval"]
    n_ahead = predict_horizon // ckpt_interval

    print(f"Trajectories: {tuple(trajs.shape)}")
    print(f"Predict {n_ahead} ckpts ahead ({predict_horizon} steps) | window K={window_k}")
    print(f"PCA dim={pca_dim} | MLP hidden={mlp_hidden} epochs={mlp_epochs} wd={mlp_wd}")
    print(f"Train={n_train} Test={N - n_train}")

    set_seed(seed)
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    print("Aligning trajectories...")
    aligned = align_trajectories(trajs, meta, H, depth)
    factory = make_model_factory(H, depth)

    # PCA on aligned (computed from TRAIN trajectories only to avoid leakage)
    train_aln = aligned[train_idx].reshape(-1, D)
    mean_aln = train_aln.mean(dim=0)
    centered = train_aln - mean_aln
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    comps_aln = Vh[:pca_dim]  # (k, D)
    total_var = (S**2).sum()
    print(f"  PCA-{pca_dim} captures {(S[:pca_dim]**2).sum() / total_var * 100:.1f}% of train variance")

    # PCA on raw (same approach)
    train_raw = trajs[train_idx].reshape(-1, D)
    mean_raw = train_raw.mean(dim=0)
    centered_raw = train_raw - mean_raw
    _, S_raw, Vh_raw = torch.linalg.svd(centered_raw, full_matrices=False)
    comps_raw = Vh_raw[:pca_dim]

    # Build window-conditioned training pairs:
    # Input: [pc(w_{t-K+1}), ..., pc(w_t)] flattened
    # Target: pc(w_{t+n_ahead})
    # t ranges from K-1 to T-1-n_ahead (need K history + n_ahead target)
    min_t = window_k - 1
    max_t = T - n_ahead - 1
    print(f"Training pairs: t in [{min_t}, {max_t}], window=[{min_t - window_k + 1}..{min_t}], target=t+{n_ahead}")

    def build_window_pairs(data, comps, mean, indices):
        """Build (X_window, Y_target) pairs from trajectory data."""
        X_list, Y_list = [], []
        for i in indices:
            for t in range(min_t, max_t + 1):
                # Window: checkpoints t-K+1, ..., t (K checkpoints)
                window_pcs = []
                for k in range(window_k):
                    pc = (data[i, t - window_k + 1 + k] - mean) @ comps.t()
                    window_pcs.append(pc)
                X_list.append(torch.cat(window_pcs))  # (K * pca_dim,)
                # Target: checkpoint t + n_ahead
                Y_list.append((data[i, t + n_ahead] - mean) @ comps.t())
        return torch.stack(X_list), torch.stack(Y_list)

    train_X_aln, train_Y_aln = build_window_pairs(aligned, comps_aln, mean_aln, train_idx)
    train_X_raw, train_Y_raw = build_window_pairs(trajs, comps_raw, mean_raw, train_idx)

    print(f"  Train pairs: {train_X_aln.shape[0]} (aligned) / {train_X_raw.shape[0]} (raw)")

    # Train MLPs
    def train_mlp(X, Y, tag, dev):
        mlp = WindowMLP(window_k, pca_dim, hidden=mlp_hidden, n_layers=2).to(dev)
        opt = torch.optim.AdamW(mlp.parameters(), lr=mlp_lr, weight_decay=mlp_wd)
        Xd, Yd = X.to(dev), Y.to(dev)
        best_loss = float("inf")
        last_loss = float("inf")
        for epoch in range(mlp_epochs):
            mlp.train()
            idx = torch.randperm(Xd.shape[0])
            for s in range(0, len(idx), 64):
                b = idx[s:s+64]
                last_loss = torch.nn.functional.mse_loss(mlp(Xd[b]), Yd[b])
                opt.zero_grad(); last_loss.backward(); opt.step()
            if last_loss.item() < best_loss:
                best_loss = last_loss.item()
        mlp.eval()
        print(f"  [{tag}] MLP best train loss: {best_loss:.6f}")
        return mlp

    print("Training predictors...")
    mlp_aln = train_mlp(train_X_aln, train_Y_aln, "aligned", device)
    mlp_raw = train_mlp(train_X_raw, train_Y_raw, "raw", device)

    # --- Evaluate ---
    arms = ["identity", "actual_future", "window_mlp_aln", "window_mlp_raw"]
    results = {a: [] for a in arms}
    per_t = {}

    print(f"\nEvaluating {len(test_idx)} test trajectories × {max_t - min_t + 1} t-values...")
    for idx_i, i in enumerate(test_idx):
        for t in range(min_t, max_t + 1):
            w_t = trajs[i, t]
            w_future = trajs[i, t + n_ahead]
            w_t_aln = aligned[i, t]
            eval_seed = seed + int(i) * 100 + t

            ce_id = eval_weights(w_t, factory, device, eval_seed)
            results["identity"].append(ce_id)
            results["actual_future"].append(eval_weights(w_future, factory, device, eval_seed))

            # Window MLP aligned
            window_aln = torch.cat([
                (aligned[i, t - window_k + 1 + k] - mean_aln) @ comps_aln.t()
                for k in range(window_k)
            ]).unsqueeze(0).to(device)
            with torch.no_grad():
                pc_pred = mlp_aln(window_aln).squeeze(0).cpu()
            pred = (pc_pred @ comps_aln) + mean_aln
            results["window_mlp_aln"].append(eval_weights(pred.reshape(-1)[:D], factory, device, eval_seed))

            # Window MLP raw
            window_raw = torch.cat([
                (trajs[i, t - window_k + 1 + k] - mean_raw) @ comps_raw.t()
                for k in range(window_k)
            ]).unsqueeze(0).to(device)
            with torch.no_grad():
                pc_pred_r = mlp_raw(window_raw).squeeze(0).cpu()
            pred_r = (pc_pred_r @ comps_raw) + mean_raw
            results["window_mlp_raw"].append(eval_weights(pred_r.reshape(-1)[:D], factory, device, eval_seed))

            if t not in per_t:
                per_t[t] = {a: [] for a in arms}
            for a in arms:
                per_t[t][a].append(results[a][-1])

        if (idx_i + 1) % 5 == 0 or idx_i == 0:
            print(f"  test traj {idx_i+1}/{len(test_idx)} done")

    # --- Summary ---
    summary = {
        "N": N, "T": T, "D": D, "hidden": H, "depth": depth,
        "predict_horizon": predict_horizon, "n_ckpts_ahead": n_ahead,
        "window_k": window_k, "pca_dim": pca_dim,
        "mlp_hidden": mlp_hidden, "mlp_epochs": mlp_epochs, "mlp_wd": mlp_wd,
        "n_train": n_train, "n_test": len(test_idx),
        "n_eval_points": len(results["identity"]),
    }
    for arm in arms:
        vals = results[arm]
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[arm] = {"ce_mean": m, "ce_std": sd, "n": len(vals)}

    id_mean = summary["identity"]["ce_mean"]
    for arm in arms:
        if arm == "identity": continue
        summary[arm]["delta_vs_identity"] = id_mean - summary[arm]["ce_mean"]

    summary["headroom"] = id_mean - summary["actual_future"]["ce_mean"]
    summary["alignment_helps"] = summary["window_mlp_aln"]["ce_mean"] < summary["window_mlp_raw"]["ce_mean"]

    per_t_summary = {}
    for t in sorted(per_t.keys()):
        per_t_summary[t] = {a: statistics.fmean(per_t[t][a]) for a in arms}
    summary["per_t_breakdown"] = per_t_summary

    predictor_arms = [a for a in arms if a not in ("identity", "actual_future")]
    best = min(predictor_arms, key=lambda a: summary[a]["ce_mean"])
    summary["best_predictor"] = best
    summary["best_delta_vs_identity"] = id_mean - summary[best]["ce_mean"]
    summary["gate_passes"] = summary["best_delta_vs_identity"] > 0.01
    summary["capture_pct"] = (
        summary["best_delta_vs_identity"] / summary["headroom"] * 100
        if summary["headroom"] > 1e-8 else 0.0
    )

    return summary


def main():
    p = argparse.ArgumentParser(description="GATE-TRAJ v3: window-conditioned trajectory prediction")
    p.add_argument("--trajectories", type=str, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--predict_horizon", type=int, default=200)
    p.add_argument("--window_k", type=int, default=3, help="Number of checkpoints in input window")
    p.add_argument("--pca_dim", type=int, default=32)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--mlp_epochs", type=int, default=300)
    p.add_argument("--mlp_wd", type=float, default=1e-2)
    p.add_argument("--out", type=str, default="checkpoints/align/traj_gate_v3")
    args = p.parse_args()

    traj_path = Path(args.trajectories)
    trajs = torch.load(traj_path, weights_only=True)
    meta = json.loads((traj_path.parent / "meta.json").read_text())

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    summary = run_window_gate(
        trajs, meta, hidden=args.hidden, depth=args.depth,
        predict_horizon=args.predict_horizon, window_k=args.window_k,
        pca_dim=args.pca_dim, n_train=args.n_train, seed=args.seed,
        device=device, mlp_hidden=args.mlp_hidden, mlp_epochs=args.mlp_epochs,
        mlp_wd=args.mlp_wd,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traj_gate_v3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== GATE-TRAJ v3 SUMMARY ===")
    print(f"Model: depth={summary['depth']} H={summary['hidden']} D={summary['D']}")
    print(f"Predict: {summary['predict_horizon']} steps ahead | Window K={summary['window_k']}")
    print(f"Train: {summary['n_train']} | Test: {summary['n_test']} | Eval points: {summary['n_eval_points']}")
    print()
    for arm in ["identity", "actual_future", "window_mlp_aln", "window_mlp_raw"]:
        s = summary[arm]
        d = s.get("delta_vs_identity", 0.0)
        ds = f"  delta={d:+.4f}" if arm != "identity" else ""
        print(f"  {arm:20s}  CE={s['ce_mean']:.4f} ± {s['ce_std']:.4f}{ds}")
    print()
    print(f"Headroom: {summary['headroom']:+.4f}")
    print(f"Capture: {summary['capture_pct']:.1f}% of headroom")
    print(f"Best: {summary['best_predictor']}  delta={summary['best_delta_vs_identity']:+.4f}")
    print(f"GATE PASSES: {summary['gate_passes']}")
    print(f"Alignment helps: {summary['alignment_helps']}")
    print("\n  Per-checkpoint CE:")
    print(f"  {'t':>3s} {'step':>5s}  {'identity':>8s} {'actual_f':>8s} {'win_aln':>8s} {'win_raw':>8s}  {'headroom':>8s} {'cap%':>6s}")
    ci = meta.get("ckpt_interval", 100)
    for t in sorted(summary["per_t_breakdown"].keys()):
        row = summary["per_t_breakdown"][t]
        hr = row["identity"] - row["actual_future"]
        cap = ((row["identity"] - row["window_mlp_aln"]) / hr * 100) if abs(hr) > 1e-8 else 0.0
        print(f"  {t:>3d} {t*ci:>5d}  {row['identity']:>8.4f} {row['actual_future']:>8.4f} {row['window_mlp_aln']:>8.4f} {row['window_mlp_raw']:>8.4f}  {hr:>+8.4f} {cap:>5.0f}%")
    print(f"\nArtifacts: {out_dir}/traj_gate_v3_summary.json")


if __name__ == "__main__":
    main()