"""
Does InfoNCE (many negatives) resist cross-modal nuisance correlation, or does it
fail as ν→1 like CA/CCA?

Linear closed-form spiked-model test — the SAME setup as Figure 3 / Section 4.1
(experiment E4 in run_experiment.py) — to isolate the OBJECTIVE from any
architecture/encoder confound. Data generator, parameters, CA/CP closed forms and
the recovered-subspace extraction are reused verbatim from src/linear/*. Only the
label-free InfoNCE method is added.

Methods (all linear encoders W,V : R^d → R^k):
  CA   — CCA closed form               (solvers.solve_ca)
  CP   — truncated RRR closed form     (solvers.solve_cp)
  InfoNCE(τ) — W,V trained by Adam on the symmetric NT-Xent loss, positives =
               paired (x_i,y_i), negatives = the other y_j (and symmetric) in a
               large batch. τ ∈ {0.05,0.1,0.5}.

Metric (as stated by the reviewer): ||P_Û − P_U||_F / sqrt(2k), P = projector onto
the k-dim subspace, U = true x-signal subspace (Q_x[:,:k]), Û = recovered subspace
(top-k input directions of the learned/closed-form encoder, via
metrics.recovered_subspace). NOTE this equals sqrt(metrics.subspace_distance), the
quantity Figure 3/E4 plots. Lower = better recovery.

    python -m src.linear.infonce_vs_ca                 # full grid
    python -m src.linear.infonce_vs_ca --smoke         # 1 point timing check
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_multimodal_data, compute_sample_covariances  # noqa: E402
from solvers import solve_ca, solve_cp                                          # noqa: E402
from metrics import recovered_subspace                                          # noqa: E402

# ── Figure 3 / E4 parameters (verbatim) ──────────────────────────────────────
D = 20
K = 3
KAPPAS = np.array([3.0, 2.0, 1.5])
GAMMA_X_S = np.ones(K) * 0.5
GAMMA_Y_S = np.ones(K) * 0.05
GAMMA_X_N = np.ones(D - K) * 1.0
GAMMA_Y_N = np.ones(D - K) * 50.0
MAX_ETA = np.sqrt(GAMMA_X_N[0] * GAMMA_Y_N[0])       # sqrt(50) ≈ 7.071
N_SAMPLES = 5000                                      # as E4

NU_GRID = np.linspace(0.0, 0.95, 15)                 # reviewer: ~15 points in [0,0.95]
N_SEEDS = 20                                          # reviewer: 20 rotation seeds
TAUS = [0.05, 0.1, 0.5]

# InfoNCE optimization
BATCH = 1024
LR = 1e-2
MAX_STEPS = 5000
PATIENCE = 300
TOL = 1e-4


def proj_subspace_distance(U, Uhat, k=K):
    """||P_Uhat - P_U||_F / sqrt(2k)  (reviewer's metric; = sqrt(1-(1/k)Σcos²θ))."""
    Q1, _ = np.linalg.qr(U)
    Q1 = Q1[:, :U.shape[1]]
    Q2, _ = np.linalg.qr(Uhat)
    Q2 = Q2[:, :Uhat.shape[1]]
    P1 = Q1 @ Q1.T
    P2 = Q2 @ Q2.T
    return float(np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2 * k))


def train_infonce(X, Y, k, tau, seed, device='cpu',
                  batch=BATCH, lr=LR, max_steps=MAX_STEPS,
                  patience=PATIENCE, tol=TOL):
    """Train linear W,V by Adam on symmetric NT-Xent. Returns W (k,d) np array."""
    import torch
    import torch.nn.functional as Fn
    g = torch.Generator(device='cpu').manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, np.float32), device=device)
    Yt = torch.tensor(np.asarray(Y, np.float32), device=device)
    n, d = Xt.shape
    W = (torch.randn(k, d, generator=g) / np.sqrt(d)).to(device).requires_grad_()
    V = (torch.randn(k, d, generator=g) / np.sqrt(d)).to(device).requires_grad_()
    opt = torch.optim.Adam([W, V], lr=lr)
    B = min(batch, n)
    labels = torch.arange(B, device=device)
    losses = []
    for step in range(max_steps):
        idx = torch.randint(0, n, (B,), generator=g).to(device)
        xb, yb = Xt[idx], Yt[idx]
        zx = xb @ W.t()
        zy = yb @ V.t()
        zx = zx / zx.norm(dim=1, keepdim=True).clamp_min(1e-8)
        zy = zy / zy.norm(dim=1, keepdim=True).clamp_min(1e-8)
        logits = zx @ zy.t() / tau
        loss = 0.5 * (Fn.cross_entropy(logits, labels) +
                      Fn.cross_entropy(logits.t(), labels))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step > 2 * patience:
            recent = np.mean(losses[-patience:])
            prev = np.mean(losses[-2 * patience:-patience])
            if abs(prev - recent) < tol:
                break
    return W.detach().cpu().numpy(), step + 1


def run_point(nu, seed, taus, device='cpu'):
    """One (ν,seed): returns dict method -> proj subspace distance."""
    etas = np.ones(D - K) * nu * MAX_ETA
    data = generate_multimodal_data(
        n=N_SAMPLES, d=D, k=K, kappas=KAPPAS,
        gamma_x_s=GAMMA_X_S, gamma_y_s=GAMMA_Y_S,
        gamma_x_n=GAMMA_X_N, gamma_y_n=GAMMA_Y_N,
        etas=etas, seed=seed)
    S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)
    U_true = data.signal_subspace_x
    out = {}
    ca = solve_ca(S_xx, S_yy, S_xy, K)
    out['CA'] = proj_subspace_distance(U_true, recovered_subspace(ca.W, S_xx))
    cp = solve_cp(S_xx, S_yy, S_xy, K)
    out['CP'] = proj_subspace_distance(U_true, recovered_subspace(cp.E, S_xx))
    for tau in taus:
        W, _ = train_infonce(data.X, data.Y, K, tau, seed=seed, device=device)
        out[f'InfoNCE_{tau}'] = proj_subspace_distance(U_true, recovered_subspace(W, S_xx))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    if args.smoke:
        t0 = time.time()
        etas = np.ones(D - K) * 0.9 * MAX_ETA
        data = generate_multimodal_data(
            n=N_SAMPLES, d=D, k=K, kappas=KAPPAS,
            gamma_x_s=GAMMA_X_S, gamma_y_s=GAMMA_Y_S,
            gamma_x_n=GAMMA_X_N, gamma_y_n=GAMMA_Y_N, etas=etas, seed=0)
        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)
        U_true = data.signal_subspace_x
        ca = solve_ca(S_xx, S_yy, S_xy, K)
        cp = solve_cp(S_xx, S_yy, S_xy, K)
        print("CA   ", proj_subspace_distance(U_true, recovered_subspace(ca.W, S_xx)))
        print("CP   ", proj_subspace_distance(U_true, recovered_subspace(cp.E, S_xx)))
        for tau in TAUS:
            tt = time.time()
            W, steps = train_infonce(data.X, data.Y, K, tau, seed=0, device=args.device)
            print(f"InfoNCE τ={tau}: dist={proj_subspace_distance(U_true, recovered_subspace(W, S_xx)):.4f} "
                  f"steps={steps} time={time.time()-tt:.1f}s")
        print(f"total smoke time {time.time()-t0:.1f}s")
        return

    cols = ['CA', 'CP'] + [f'InfoNCE_{t}' for t in TAUS]
    means = {c: [] for c in cols}
    stds = {c: [] for c in cols}
    t0 = time.time()
    for nu in NU_GRID:
        per = {c: [] for c in cols}
        for seed in range(N_SEEDS):
            r = run_point(nu, seed, TAUS, device=args.device)
            for c in cols:
                per[c].append(r[c])
        for c in cols:
            means[c].append(np.mean(per[c]))
            stds[c].append(np.std(per[c]))
        print(f"# ν={nu:.3f} done  ({time.time()-t0:.0f}s)", file=sys.stderr)

    lines = []
    lines.append("## InfoNCE vs CA/CP — subspace recovery vs nuisance correlation ν "
                 "(linear spiked model, Figure-3 setup)\n")
    lines.append(f"Metric = ‖P_Û−P_U‖_F / √(2k), mean [±std] over {N_SEEDS} seeds. "
                 "Lower = better recovery. ε not applicable (closed-form / GD).\n")
    header = "| ν | " + " | ".join(
        ['CA', 'CP', 'InfoNCE(τ=0.05)', 'InfoNCE(τ=0.1)', 'InfoNCE(τ=0.5)']) + " |"
    lines.append(header)
    lines.append("|--:|" + "|".join(["--"] * 5) + "|")
    for i, nu in enumerate(NU_GRID):
        cells = []
        for c in cols:
            cells.append(f"{means[c][i]:.3f} [±{stds[c][i]:.3f}]")
        lines.append(f"| {nu:.3f} | " + " | ".join(cells) + " |")

    # crossing points (first ν where mean metric > 0.5)
    lines.append("\n**Failure point** (first ν where mean subspace distance crosses 0.5):\n")
    for c, lbl in zip(cols, ['CA', 'CP', 'InfoNCE(τ=0.05)', 'InfoNCE(τ=0.1)', 'InfoNCE(τ=0.5)']):
        arr = np.array(means[c])
        cross = np.where(arr > 0.5)[0]
        msg = f"ν ≈ {NU_GRID[cross[0]]:.3f}" if cross.size else "never crosses 0.5 in [0,0.95]"
        lines.append(f"- {lbl}: {msg}")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(out)
    print(f"# total {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == '__main__':
    main()
