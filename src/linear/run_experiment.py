"""
Main experiment runner for spiked synthetic experiments.

Runs 8 controlled experiments (E1–E8) that verify the paper's
theoretical results. Each experiment sweeps one parameter and
compares empirical measurements against closed-form predictions.

Usage:
    python src/linear/run_experiment.py              # run all experiments
    python src/linear/run_experiment.py --exp E1     # run only E1
    python src/linear/run_experiment.py --exp E1 E3  # run E1 and E3
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_data import generate_multimodal_data, compute_sample_covariances
from solvers import solve_ca, solve_cp, solve_ca_probe, solve_joint
from theory import (
    compute_recovery_predictions,
    compute_soft_bottleneck_bound,
    compute_separation_ratio_formula,
    compute_ca_canonical_correlations,
)
from metrics import (
    subspace_distance,
    prediction_error,
    encoder_norm_pair,
    recovered_subspace,
)

# ── Plotting defaults ──────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Default parameters ─────────────────────────────────────────────
N_SAMPLES = 5000
N_SEEDS = 10
D_DEFAULT = 20
K_DEFAULT = 3


def _run_over_seeds(sweep_values, run_one_fn, n_seeds=N_SEEDS):
    """
    Run an experiment over multiple seeds for each sweep value.

    run_one_fn(value, seed) -> dict of metric_name -> float
    Returns dict of metric_name -> (means, stds) arrays.
    """
    all_results = {}
    for val in sweep_values:
        seed_results = {}
        for seed in range(n_seeds):
            result = run_one_fn(val, seed)
            for key, v in result.items():
                seed_results.setdefault(key, []).append(v)
        for key, vals in seed_results.items():
            means = all_results.setdefault(key + "_mean", [])
            stds = all_results.setdefault(key + "_std", [])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    return {k: np.array(v) for k, v in all_results.items()}


def _eigh_inv_sqrt(M):
    """Compute M^{-1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ═══════════════════════════════════════════════════════════════════
# E1: Hard Bottleneck (Proposition 2.1)
# ═══════════════════════════════════════════════════════════════════
def experiment_e1():
    """
    Test: max recoverable shared dimensions = min(r_x, r_y, r_xy).

    Sweep 1: effective rank r_xy from 1 to k (Constraining interaction).
    Sweep 2: effective rank r_y from 1 to k (Constraining marginal).
    Use POPULATION covariances so ranks are exact.
    Measure: number of non-zero canonical correlations.
    """
    print("\n" + "=" * 60)
    print("E1: Hard Bottleneck (Proposition 2.1)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappas_base = np.array([3.0, 2.5, 2.0])

    rank_values = np.arange(1, k + 1)

    def run_one(rank_val, mode, seed):
        # Default full rank setup
        kappas_eff = kappas_base.copy()
        gamma_x_s = np.ones(k) * 0.5
        gamma_y_s = np.ones(k) * 0.5
        gamma_x_n = np.ones(d - k) * 1.0
        gamma_y_n = np.ones(d - k) * 1.0
        etas = np.zeros(d - k)

        if mode == "r_xy":
            # Constraint: r_xy = rank_val
            # We zero out the smallest signal strengths (kappas) beyond rank_val
            # AND strictly zero out etas (which are already 0)
            kappas_eff[rank_val:] = 0.0
            
        elif mode == "r_y":
            # Constraint: r_y = rank_val
            # We must make S_yy rank deficient.
            # S_yy diag = [kappas^2 + gamma_s, gamma_n]
            # To limit rank to `rank_val` (where rank_val <= k):
            # 1. Keep top `rank_val` signal dimensions active.
            # 2. Zero out remaining signal dimensions (set kappas=0, gamma_s=0).
            # 3. Zero out ALL noise dimensions (gamma_n=0).
            
            # Disable signals beyond rank_val
            kappas_eff[rank_val:] = 0.0
            gamma_y_s[rank_val:] = 0.0
            
            # Disable all nuisance noise to ensure r_y <= k
            gamma_y_n[:] = 0.0
            
            # Note: For r_x we leave it full rank.

        rng = np.random.default_rng(seed)
        Q_x = _random_ortho(d, rng)
        Q_y = _random_ortho(d, rng)

        # Build POPULATION covariances directly
        S_xx_diag = np.concatenate([kappas_eff**2 + gamma_x_s, gamma_x_n])
        S_yy_diag = np.concatenate([kappas_eff**2 + gamma_y_s, gamma_y_n])
        S_xy_diag = np.concatenate([kappas_eff**2, etas])

        S_xx = Q_x @ np.diag(S_xx_diag) @ Q_x.T
        S_yy = Q_y @ np.diag(S_yy_diag) @ Q_y.T
        S_xy = Q_x @ np.diag(S_xy_diag) @ Q_y.T

        # For S_yy singular, add epsilon for numerical stability of inversion if solver requires it
        # But solve_ca uses generalized eigen problem. Let's rely on pinv logic inside solver or add tiny jitter.
        # Adding tiny jitter to S_yy diagonals for stability (effectively full rank but tiny eigenvalues)
        # would defeat the purpose of "hard" rank test? 
        # Actually our solvers use eigh which is robust. 
        # But let's add minimal regularization if needed or trust the math.
        # If r_y < k, S_yy is singular. solve_ca computes S_yy^{-1/2}.
        # _eigh_inv_sqrt caps eigenvalues at 1e-12. This effectively "inverts" 0 to large number.
        # This is correct for CCA: 0 variance directions are excluded or explode.
        # Let's see if it works.

        ca_sol = solve_ca(S_xx, S_yy, S_xy, k)

        n_recovered = np.sum(ca_sol.canonical_correlations > 1e-6)
        return {"n_recovered": n_recovered}

    # Run sweeps
    results_r_xy = _run_over_seeds(rank_values, lambda r, s: run_one(r, "r_xy", s))
    results_r_y  = _run_over_seeds(rank_values, lambda r, s: run_one(r, "r_y", s))

    fig, ax = plt.subplots()
    
    # Theory line
    ax.plot(rank_values, rank_values, "k--", label="Theory: $\\min(r_x, r_y, r_{xy})$", linewidth=2)
    
    # r_xy results
    ax.plot(rank_values, results_r_xy["n_recovered_mean"], "o-", color="tab:blue", label="Varying $r_{xy}$ (fixed $r_x, r_y$ full)")
    ax.fill_between(
        rank_values,
        results_r_xy["n_recovered_mean"] - results_r_xy["n_recovered_std"],
        results_r_xy["n_recovered_mean"] + results_r_xy["n_recovered_std"],
        alpha=0.2, color="tab:blue",
    )

    # r_y results
    ax.plot(rank_values, results_r_y["n_recovered_mean"], "s-", color="tab:orange", label="Varying $r_y$ (fixed $r_{xy}=k$)")
    ax.fill_between(
        rank_values,
        results_r_y["n_recovered_mean"] - results_r_y["n_recovered_std"],
        results_r_y["n_recovered_mean"] + results_r_y["n_recovered_std"],
        alpha=0.2, color="tab:orange",
    )

    ax.set_xlabel("Constraining Rank Value")
    ax.set_ylabel("Recovered shared dimensions")
    ax.set_title("Hard Bottleneck: $r_{xy}$ vs $r_y$")
    ax.legend()
    ax.set_xticks(rank_values)
    ax.set_yticks(rank_values)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E1_hard_bottleneck.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E1_hard_bottleneck.png")
    return results_r_xy


def _random_ortho(d, rng):
    """Random orthogonal matrix."""
    H = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(H)
    return Q @ np.diag(np.sign(np.diag(R)))


# ═══════════════════════════════════════════════════════════════════
# E1_multi: Hard Bottleneck for N Modalities
# ═══════════════════════════════════════════════════════════════════
def experiment_e1_multimodal():
    """
    N-modality extension of E1 (Proposition 2.1).  Two panels:

    (a) Single bottleneck: one rank-r modality, rest full rank.
        Sweep r=1..k, show N=2,3,5 all track theory min_i(r_i).

    (b) Cascading ranks: k=5, modalities added with ranks k, k-1, ..., 1.
        Sweep N=2..k, show recovery = k-N+1 (monotone decrease).
    """
    print("\n" + "=" * 60)
    print("E1_multi: Hard Bottleneck — N Modalities")
    print("=" * 60)

    # ── shared helpers ──────────────────────────────────────────────
    def _pop_cov(kappas_eff, gamma_s, gamma_n, Q):
        d = Q.shape[0]
        k = len(kappas_eff)
        diag = np.concatenate([kappas_eff**2 + gamma_s, gamma_n])
        return Q @ np.diag(diag) @ Q.T

    def _cross_cov(kappas_i, kappas_j, Q_i, Q_j):
        d = Q_i.shape[0]
        k = len(kappas_i)
        cross_diag = np.concatenate([kappas_i * kappas_j, np.zeros(d - k)])
        return Q_i @ np.diag(cross_diag) @ Q_j.T

    def _pairwise_intersection(S_list, kappas_list, Qs, k_max):
        """Count shared dims via pairwise CCA with anchor (mod 0)."""
        N = len(S_list)
        n_shared = k_max
        for j in range(1, N):
            S_0j = _cross_cov(kappas_list[0], kappas_list[j], Qs[0], Qs[j])
            ca_sol = solve_ca(S_list[0], S_list[j], S_0j, k_max)
            n_j = int(np.sum(ca_sol.canonical_correlations > 1e-6))
            n_shared = min(n_shared, n_j)
        return n_shared

    # ── Panel (a): single bottleneck, vary N ────────────────────────
    d = D_DEFAULT
    k = K_DEFAULT
    kappas_base = np.array([3.0, 2.5, 2.0])
    gamma_s = np.ones(k) * 0.5
    gamma_n = np.ones(d - k) * 1.0
    rank_values = np.arange(1, k + 1)
    N_list = [2, 3, 5]

    def run_single_bottleneck(r_bottle, N, seed):
        rng = np.random.default_rng(seed)
        Qs = [_random_ortho(d, rng) for _ in range(N)]
        kappas_list = [kappas_base.copy() for _ in range(N)]
        kappas_list[1][r_bottle:] = 0.0  # modality 1 is the bottleneck
        S_list = [_pop_cov(kappas_list[i], gamma_s, gamma_n, Qs[i]) for i in range(N)]
        return {"n_recovered": _pairwise_intersection(S_list, kappas_list, Qs, k)}

    # ── Panel (b): cascading ranks, vary N ──────────────────────────
    k_c = 5
    d_c = 30
    kappas_c = np.array([3.0, 2.8, 2.5, 2.2, 2.0])
    gamma_s_c = np.ones(k_c) * 0.5
    gamma_n_c = np.ones(d_c - k_c) * 1.0
    N_values = np.arange(2, k_c + 1)  # 2..5

    def run_cascading(N, seed):
        rng = np.random.default_rng(seed)
        Qs = [_random_ortho(d_c, rng) for _ in range(N)]
        kappas_list = []
        for i in range(N):
            keff = kappas_c.copy()
            keff[k_c - i:] = 0.0  # modality i has rank k_c-i
            kappas_list.append(keff)
        S_list = [_pop_cov(kappas_list[i], gamma_s_c, gamma_n_c, Qs[i]) for i in range(N)]
        return {"n_recovered": _pairwise_intersection(S_list, kappas_list, Qs, k_c)}

    # ── Run cascading sweep ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4.5))

    theory_cascade = k_c + 1 - N_values
    ax.plot(N_values, theory_cascade, "k--", lw=2, label=r"Theory: $k - N + 1$")
    res_c = _run_over_seeds(N_values, lambda N, s: run_cascading(N, s))
    ax.plot(N_values, res_c["n_recovered_mean"], "o-", color="tab:purple", label="Empirical")
    ax.fill_between(N_values,
                    res_c["n_recovered_mean"] - res_c["n_recovered_std"],
                    res_c["n_recovered_mean"] + res_c["n_recovered_std"],
                    alpha=0.2, color="tab:purple")
    ax.set_xlabel("Number of modalities $N$")
    ax.set_ylabel("Recovered shared dimensions")
    ax.set_title(r"Hard Bottleneck: Cascading Ranks ($r_i = k{-}i{+}1$)")
    ax.set_xticks(N_values)
    ax.set_yticks(np.arange(1, k_c))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E1_multimodal_bottleneck.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E1_multimodal_bottleneck.png")


# ═══════════════════════════════════════════════════════════════════
# E2: Soft Bottleneck (Proposition 2.3)
# ═══════════════════════════════════════════════════════════════════
def experiment_e2():
    """
    Test: encoder norm >= 2/phi_k as phi_k -> 0.

    Sweep: weakest signal strength kappa_k from 0.05 to 5.
    Fix: other signal strengths at 3.0 and 2.5.
    Measure: ||W'||_F^2 + ||V'||_F^2 vs lower bound 2/phi_k.
    """
    print("\n" + "=" * 60)
    print("E2: Soft Bottleneck (Proposition 2.3)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappa_k_values = np.logspace(np.log10(0.05), np.log10(5.0), 20)

    def run_one(kappa_k, seed):
        kappas = np.array([3.0, 2.5, kappa_k])
        gamma_x_s = np.ones(k) * 0.5
        gamma_y_s = np.ones(k) * 0.5
        gamma_x_n = np.ones(d - k) * 1.0
        gamma_y_n = np.ones(d - k) * 1.0
        etas = np.zeros(d - k)

        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed,
        )

        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)
        ca_sol = solve_ca(S_xx, S_yy, S_xy, k)

        norm = encoder_norm_pair(ca_sol.W, ca_sol.V, S_xx, S_yy)
        phi_k = ca_sol.canonical_correlations[-1]
        theory_bound = compute_soft_bottleneck_bound(phi_k)

        return {"encoder_norm": norm, "theory_bound": theory_bound, "phi_k": phi_k}

    results = _run_over_seeds(kappa_k_values, run_one)

    fig, ax = plt.subplots()
    ax.semilogy(
        kappa_k_values, results["encoder_norm_mean"], "o-", color="tab:blue",
        label="Empirical $\\|W'\\|_F^2 + \\|V'\\|_F^2$",
    )
    ax.fill_between(
        kappa_k_values,
        results["encoder_norm_mean"] - results["encoder_norm_std"],
        results["encoder_norm_mean"] + results["encoder_norm_std"],
        alpha=0.2, color="tab:blue",
    )
    ax.semilogy(
        kappa_k_values, results["theory_bound_mean"], "s--", color="tab:red",
        label="Theory: $2/\\phi_k$",
    )
    ax.set_xlabel("$\\kappa_k$ (weakest signal strength)")
    ax.set_ylabel("Encoder norm (log scale)")
    ax.set_title("Soft Bottleneck")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E2_soft_bottleneck.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E2_soft_bottleneck.png")
    return results


# ═══════════════════════════════════════════════════════════════════
# E3: CA vs CP Separation (Corollary 3.5)
# ═══════════════════════════════════════════════════════════════════
def experiment_e3():
    """
    Test: CA has wider recovery regime when target noise is large.

    Sweep: gamma_y_n (target nuisance noise) from 0.1 to 100.
    Fix: moderate cross-correlation eta, fixed signal.
    Measure: Δ_CA and Δ_CP (empirical and theoretical).
    """
    print("\n" + "=" * 60)
    print("E3: CA vs CP Separation (Corollary 3.5)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappas = np.array([3.0, 2.5, 2.0])
    gamma_y_n_values = np.logspace(np.log10(0.1), np.log10(100.0), 25)
    eta_base = 0.5

    def run_one(gamma_y_n_val, seed):
        gamma_x_s = np.ones(k) * 0.5
        gamma_y_s = np.ones(k) * 0.5
        gamma_x_n = np.ones(d - k) * 1.0
        gamma_y_n = np.ones(d - k) * gamma_y_n_val

        max_eta = np.sqrt(gamma_x_n * gamma_y_n)
        etas = np.minimum(np.ones(d - k) * eta_base, max_eta * 0.99)

        preds = compute_recovery_predictions(
            kappas, gamma_x_s, gamma_y_s, gamma_x_n, gamma_y_n, etas
        )

        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed,
        )

        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)

        S_xx_inv_sqrt = _eigh_inv_sqrt(S_xx)
        S_yy_inv_sqrt = _eigh_inv_sqrt(S_yy)
        C_emp = S_xx_inv_sqrt @ S_xy @ S_yy_inv_sqrt
        sv_ca = np.linalg.svd(C_emp, compute_uv=False)

        S_yx = S_xy.T
        A_emp = S_yx @ S_xx_inv_sqrt
        sv_cp = np.linalg.svd(A_emp, compute_uv=False)

        delta_ca_emp = sv_ca[k - 1] / sv_ca[k] if sv_ca[k] > 1e-15 else np.inf
        delta_cp_emp = sv_cp[k - 1] / sv_cp[k] if sv_cp[k] > 1e-15 else np.inf

        return {
            "delta_ca_theory": preds.delta_ca,
            "delta_cp_theory": preds.delta_cp,
            "delta_ca_emp": min(delta_ca_emp, 50),  # cap for plotting
            "delta_cp_emp": min(delta_cp_emp, 50),
        }

    results = _run_over_seeds(gamma_y_n_values, run_one)

    fig, ax = plt.subplots()
    ax.semilogx(gamma_y_n_values, results["delta_ca_theory_mean"], "s--", color="tab:blue", label="$\\Delta_{CA}$ (theory)")
    ax.semilogx(gamma_y_n_values, results["delta_cp_theory_mean"], "^--", color="tab:red", label="$\\Delta_{CP}$ (theory)")
    ax.semilogx(gamma_y_n_values, results["delta_ca_emp_mean"], "o-", color="tab:blue", alpha=0.7, label="$\\Delta_{CA}$ (empirical)")
    ax.fill_between(gamma_y_n_values,
                     results["delta_ca_emp_mean"] - results["delta_ca_emp_std"],
                     results["delta_ca_emp_mean"] + results["delta_ca_emp_std"],
                     alpha=0.2, color="tab:blue")
    ax.semilogx(gamma_y_n_values, results["delta_cp_emp_mean"], "v-", color="tab:red", alpha=0.7, label="$\\Delta_{CP}$ (empirical)")
    ax.fill_between(gamma_y_n_values,
                     results["delta_cp_emp_mean"] - results["delta_cp_emp_std"],
                     results["delta_cp_emp_mean"] + results["delta_cp_emp_std"],
                     alpha=0.2, color="tab:red")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Recovery threshold")
    ax.set_xlabel("$\\tilde{\\gamma}^y$ (target nuisance noise)")
    ax.set_ylabel("Separation ratio $\\Delta$")
    ax.set_title("CA vs CP Separation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E3_ca_vs_cp_separation.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E3_ca_vs_cp_separation.png")
    return results


# ═══════════════════════════════════════════════════════════════════
# E4: Noise Correlation Threshold (Props 3.1 & 3.3)
# ═══════════════════════════════════════════════════════════════════
def experiment_e4():
    """
    Test: CA is more robust than CP as noise correlation eta increases.

    Sweep: eta from 0 to near maximum.
    Key: very small γ_y_s (signal coords clean) but very large γ̃_y (nuisance noisy).
    This is the regime where CA normalization by S_yy helps CA but not CP.
    Δ_CA/Δ_CP = √(γ̃_y / (κ_k² + γ_y_s)) ≈ 4.66  (strong separation)
    """
    print("\n" + "=" * 60)
    print("E4: Noise Correlation Threshold (Props 3.1 & 3.3)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappas = np.array([3.0, 2.0, 1.5])
    gamma_x_s = np.ones(k) * 0.5   # moderate noise on x-signal coords
    gamma_y_s = np.ones(k) * 0.05  # VERY LOW noise on y-signal coords (key!)
    gamma_x_n = np.ones(d - k) * 1.0
    gamma_y_n = np.ones(d - k) * 50.0  # VERY HIGH nuisance noise in y (key!)

    max_eta = np.sqrt(gamma_x_n[0] * gamma_y_n[0])  # sqrt(50) ≈ 7.07
    nu_values = np.linspace(0, 0.95, 30)  # normalized correlation ν ∈ [0, 1]
    eta_values = nu_values * max_eta       # convert to raw covariance for simulation

    def run_one(eta_val, seed):
        etas = np.ones(d - k) * eta_val

        preds = compute_recovery_predictions(
            kappas, gamma_x_s, gamma_y_s, gamma_x_n, gamma_y_n, etas
        )

        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed,
        )

        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)

        ca_sol = solve_ca(S_xx, S_yy, S_xy, k)
        cp_sol = solve_cp(S_xx, S_yy, S_xy, k)

        ca_subspace = recovered_subspace(ca_sol.W, S_xx)
        cp_subspace = recovered_subspace(cp_sol.E, S_xx)

        dist_ca = subspace_distance(data.signal_subspace_x, ca_subspace)
        dist_cp = subspace_distance(data.signal_subspace_x, cp_subspace)

        return {
            "dist_ca": dist_ca,
            "dist_cp": dist_cp,
            "delta_ca": min(preds.delta_ca, 50),
            "delta_cp": min(preds.delta_cp, 50),
        }

    results = _run_over_seeds(eta_values, run_one)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(nu_values, results["dist_ca_mean"], "o-", color="tab:blue", label="CA")
    ax1.fill_between(nu_values,
                      results["dist_ca_mean"] - results["dist_ca_std"],
                      results["dist_ca_mean"] + results["dist_ca_std"],
                      alpha=0.2, color="tab:blue")
    ax1.plot(nu_values, results["dist_cp_mean"], "v-", color="tab:red", label="CP")
    ax1.fill_between(nu_values,
                      results["dist_cp_mean"] - results["dist_cp_std"],
                      results["dist_cp_mean"] + results["dist_cp_std"],
                      alpha=0.2, color="tab:red")
    ax1.set_xlabel("$\\nu$ (normalized noise correlation)")
    ax1.set_ylabel("Subspace distance")
    ax1.set_title("Recovery vs Noise Correlation")
    ax1.legend()

    ax2.plot(nu_values, results["delta_ca_mean"], "o-", color="tab:blue", label="$\\Delta_{CA}$")
    ax2.plot(nu_values, results["delta_cp_mean"], "v-", color="tab:red", label="$\\Delta_{CP}$")
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Threshold")
    ax2.set_yscale("log")
    ax2.set_xlabel("$\\nu$ (normalized noise correlation)")
    ax2.set_ylabel("Separation ratio $\\Delta$")
    ax2.set_title("Separation Ratios")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E4_noise_correlation.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E4_noise_correlation.png")
    return results


# ═══════════════════════════════════════════════════════════════════
# E5: CA+Probe vs Direct CP (Proposition 3.8)
# ═══════════════════════════════════════════════════════════════════
def experiment_e5():
    """
    Test: CA+Probe outperforms CP when Δ_CA > 1 > Δ_CP.

    Sweep: eta from 0 to max.
    Same parameter regime as E4 (very small γ_y_s, very large γ̃_y) so that
    there is a wide Δ_CA > 1 > Δ_CP gap.
    Measure: subspace distance, signal MSE, and total MSE.
    The key insight: CP can achieve lower total MSE by fitting noise
    correlations, while recovering the wrong subspace entirely.
    """
    print("\n" + "=" * 60)
    print("E5: CA+Probe vs Direct CP (Proposition 3.8)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappas = np.array([3.0, 2.0, 1.5])
    gamma_x_s = np.ones(k) * 0.5
    gamma_y_s = np.ones(k) * 0.05  # VERY LOW noise on y-signal coords
    gamma_x_n = np.ones(d - k) * 1.0
    gamma_y_n = np.ones(d - k) * 50.0  # VERY HIGH nuisance noise in y

    max_eta = np.sqrt(gamma_x_n[0] * gamma_y_n[0])  # sqrt(50) ≈ 7.07
    nu_values = np.linspace(0, 0.95, 30)  # normalized correlation ν ∈ [0, 1]
    eta_values = nu_values * max_eta       # convert to raw covariance for simulation

    def run_one(eta_val, seed):
        etas = np.ones(d - k) * eta_val

        preds = compute_recovery_predictions(
            kappas, gamma_x_s, gamma_y_s, gamma_x_n, gamma_y_n, etas
        )

        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed,
        )

        data_test = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed + 1000,
            Q_x=data.Q_x, Q_y=data.Q_y,
        )

        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)

        # CP
        cp_sol = solve_cp(S_xx, S_yy, S_xy, k)
        mse_cp = prediction_error(cp_sol.B, data_test.X, data_test.Y)
        mse_cp_signal = prediction_error(cp_sol.B, data_test.X, data_test.Y_signal)
        cp_subspace = recovered_subspace(cp_sol.E, S_xx)
        dist_cp = subspace_distance(data.signal_subspace_x, cp_subspace)

        # CA + Probe
        ca_sol = solve_ca(S_xx, S_yy, S_xy, k)
        B_probe = solve_ca_probe(ca_sol.W, S_xx, S_xy)
        B_full = B_probe @ ca_sol.W
        mse_ca_probe = prediction_error(B_full, data_test.X, data_test.Y)
        mse_ca_probe_signal = prediction_error(B_full, data_test.X, data_test.Y_signal)
        ca_subspace = recovered_subspace(ca_sol.W, S_xx)
        dist_ca = subspace_distance(data.signal_subspace_x, ca_subspace)

        return {
            "mse_cp": mse_cp,
            "mse_cp_signal": mse_cp_signal,
            "mse_ca_probe": mse_ca_probe,
            "mse_ca_probe_signal": mse_ca_probe_signal,
            "dist_cp": dist_cp,
            "dist_ca": dist_ca,
            "delta_ca": min(preds.delta_ca, 50),
            "delta_cp": min(preds.delta_cp, 50),
        }

    results = _run_over_seeds(eta_values, run_one)

    # --- 2-panel figure: signal MSE | total MSE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Helper to shade the Δ_CA > 1 > Δ_CP region
    def _shade_gap(ax):
        for i, nu in enumerate(nu_values):
            if results["delta_ca_mean"][i] > 1 and results["delta_cp_mean"][i] < 1:
                dx = (nu_values[1] - nu_values[0]) / 2 if len(nu_values) > 1 else 0.01
                ax.axvspan(nu - dx, nu + dx, alpha=0.08, color="green")

    # Panel 1: Signal-only MSE
    ax1.plot(nu_values, results["mse_ca_probe_signal_mean"], "o-", color="tab:blue", label="CA + Probe")
    ax1.fill_between(nu_values,
                      results["mse_ca_probe_signal_mean"] - results["mse_ca_probe_signal_std"],
                      results["mse_ca_probe_signal_mean"] + results["mse_ca_probe_signal_std"],
                      alpha=0.2, color="tab:blue")
    ax1.plot(nu_values, results["mse_cp_signal_mean"], "v-", color="tab:red", label="CP")
    ax1.fill_between(nu_values,
                      results["mse_cp_signal_mean"] - results["mse_cp_signal_std"],
                      results["mse_cp_signal_mean"] + results["mse_cp_signal_std"],
                      alpha=0.2, color="tab:red")
    _shade_gap(ax1)
    ax1.set_xlabel("$\\nu$ (normalized noise correlation)")
    ax1.set_ylabel("Signal prediction MSE")
    ax1.set_title("(a) Signal Prediction Error")
    ax1.legend(fontsize=9)

    # Panel 2: Total MSE
    ax2.plot(nu_values, results["mse_ca_probe_mean"], "o-", color="tab:blue", label="CA + Probe")
    ax2.fill_between(nu_values,
                      results["mse_ca_probe_mean"] - results["mse_ca_probe_std"],
                      results["mse_ca_probe_mean"] + results["mse_ca_probe_std"],
                      alpha=0.2, color="tab:blue")
    ax2.plot(nu_values, results["mse_cp_mean"], "v-", color="tab:red", label="CP")
    ax2.fill_between(nu_values,
                      results["mse_cp_mean"] - results["mse_cp_std"],
                      results["mse_cp_mean"] + results["mse_cp_std"],
                      alpha=0.2, color="tab:red")
    _shade_gap(ax2)
    ax2.set_xlabel("$\\nu$ (normalized noise correlation)")
    ax2.set_ylabel("Total prediction MSE")
    ax2.set_title("(b) Total Prediction Error")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E5_ca_probe_vs_cp.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E5_ca_probe_vs_cp.png")
    return results


# ═══════════════════════════════════════════════════════════════════
# E6: Joint Objective Interpolation (Corollary 3.12)
# ═══════════════════════════════════════════════════════════════════
def experiment_e6():
    """
    Test: joint objective interpolates between CP (λ=0) and CCA (λ→∞).

    Sweep: lambda from 0 to 100.
    Fix: η=2.0 which is ABOVE CP threshold (~1.89) but BELOW CA threshold (~2.94).
    This means CP and CCA select genuinely different subspaces:
    CP picks some noise directions, CCA picks signal directions.
    The joint objective should smoothly transition from CP→CCA as λ grows.
    """
    print("\n" + "=" * 60)
    print("E6: Joint Objective Interpolation (Corollary 3.12)")
    print("=" * 60)

    d = D_DEFAULT
    k = K_DEFAULT
    kappas = np.array([3.0, 2.5, 2.0])
    gamma_x_s = np.ones(k) * 0.5
    gamma_y_s = np.ones(k) * 0.1
    gamma_x_n = np.ones(d - k) * 1.0
    gamma_y_n = np.ones(d - k) * 10.0
    etas = np.ones(d - k) * 2.0  # above CP threshold, below CA threshold

    lam_values = np.concatenate([[0], np.logspace(-2, 2, 30)])

    def run_one(lam, seed):
        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=kappas,
            gamma_x_s=gamma_x_s, gamma_y_s=gamma_y_s,
            gamma_x_n=gamma_x_n, gamma_y_n=gamma_y_n,
            etas=etas, seed=seed,
        )

        S_xx, S_yy, S_xy = compute_sample_covariances(data.X, data.Y)

        ca_sol = solve_ca(S_xx, S_yy, S_xy, k)
        cp_sol = solve_cp(S_xx, S_yy, S_xy, k)
        ca_subspace = recovered_subspace(ca_sol.W, S_xx)
        cp_subspace = recovered_subspace(cp_sol.E, S_xx)

        joint_sol = solve_joint(S_xx, S_yy, S_xy, k, lam)
        joint_subspace = recovered_subspace(joint_sol.W, S_xx)

        dist_to_cca = subspace_distance(ca_subspace, joint_subspace)
        dist_to_cp = subspace_distance(cp_subspace, joint_subspace)
        dist_to_true = subspace_distance(data.signal_subspace_x, joint_subspace)

        return {
            "dist_to_cca": dist_to_cca,
            "dist_to_cp": dist_to_cp,
            "dist_to_true": dist_to_true,
        }

    results = _run_over_seeds(lam_values, run_one)

    fig, ax = plt.subplots()
    # Skip λ=0 for log scale on x-axis
    ax.semilogx(lam_values[1:], results["dist_to_cca_mean"][1:], "o-", color="tab:blue", label="Distance to CCA solution")
    ax.fill_between(lam_values[1:],
                     results["dist_to_cca_mean"][1:] - results["dist_to_cca_std"][1:],
                     results["dist_to_cca_mean"][1:] + results["dist_to_cca_std"][1:],
                     alpha=0.2, color="tab:blue")
    ax.semilogx(lam_values[1:], results["dist_to_cp_mean"][1:], "v-", color="tab:red", label="Distance to CP solution")
    ax.fill_between(lam_values[1:],
                     results["dist_to_cp_mean"][1:] - results["dist_to_cp_std"][1:],
                     results["dist_to_cp_mean"][1:] + results["dist_to_cp_std"][1:],
                     alpha=0.2, color="tab:red")

    ax.axvline(x=lam_values[1], color="gray", linestyle=":", alpha=0.3)
    ax.annotate(f"$\\lambda=0$: dist_CP={results['dist_to_cp_mean'][0]:.3f}",
                xy=(lam_values[1], results["dist_to_cp_mean"][0]), fontsize=9, color="tab:red")

    ax.set_xlabel("$\\lambda$ (alignment weight)")
    ax.set_ylabel("Subspace distance")
    ax.set_title("Joint Objective Interpolation CP ↔ CCA")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E6_joint_interpolation.png"), dpi=150)
    plt.close(fig)
    print("  → Saved E6_joint_interpolation.png")
    return results


# ═══════════════════════════════════════════════════════════════════
# E7: Estimation Validation (κ̂, ν̂)
# ═══════════════════════════════════════════════════════════════════

def _pls_decomposition(X, Y, n_components):
    """PLS via SVD of cross-covariance matrix."""
    n = X.shape[0]
    X_c = X - X.mean(0)
    Y_c = Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(X_c.T @ Y_c / n, full_matrices=False)
    k = min(n_components, len(s))
    return U[:, :k], Vt[:k].T, s[:k]


def _cca_decomposition(X, Y, n_components):
    """CCA via whitened SVD of cross-covariance matrix."""
    n = X.shape[0]
    X_c = X - X.mean(0)
    Y_c = Y - Y.mean(0)
    Sxx = X_c.T @ X_c / n
    Syy = Y_c.T @ Y_c / n
    Sxy = X_c.T @ Y_c / n
    Sx_isqrt = _eigh_inv_sqrt(Sxx)
    Sy_isqrt = _eigh_inv_sqrt(Syy)
    M = Sx_isqrt @ Sxy @ Sy_isqrt
    P, sv, Qt = np.linalg.svd(M, full_matrices=False)
    k = min(n_components, len(sv))
    U_cca = Sx_isqrt @ P[:, :k]
    return U_cca, np.clip(sv[:k], 0, 1)


def _estimate_kappa_nu(X, Y, signal, n_components, epsilon=0.05):
    """
    Estimate (kappa_hat, nu_hat) from paired multimodal data.

    Adapted from MultiDESA's analyze_phase_diagram.py:
      kappa_hat = PLS SV at first component where cumulative R^2 > epsilon
      nu_hat    = max CCA canonical correlation among nuisance components

    Parameters
    ----------
    X, Y : (n, d) arrays — paired multimodal observations
    signal : (n, k_signal) array — ground truth shared signal
    n_components : int — number of components to extract
    epsilon : float — R^2 threshold for signal/nuisance classification

    Returns
    -------
    kappa_hat, nu_hat : floats
    """
    nc = min(n_components, X.shape[1], Y.shape[1])
    X_c = X - X.mean(0)
    Y_c = Y - Y.mean(0)
    sig_c = signal - signal.mean(0)
    ss_tot = np.sum(sig_c ** 2)
    if ss_tot < 1e-15:
        return 0.0, 0.0

    # ── PLS: estimate κ̂ ──
    U_pls, V_pls, pls_svs = _pls_decomposition(X, Y, nc)
    cum_r2 = np.zeros(len(pls_svs))
    for i in range(len(pls_svs)):
        Z = np.hstack([X_c @ U_pls[:, :i + 1], Y_c @ V_pls[:, :i + 1]])
        beta = np.linalg.lstsq(Z, sig_c, rcond=None)[0]
        ss_res = np.sum((sig_c - Z @ beta) ** 2)
        cum_r2[i] = max(1.0 - ss_res / ss_tot, 0.0)

    above = cum_r2 > epsilon
    kappa_hat = float(pls_svs[np.where(above)[0][0]]) if above.any() else 0.0

    # ── CCA: estimate ν̂ ──
    U_cca, canon_corrs = _cca_decomposition(X, Y, nc)
    per_r2 = np.zeros(len(canon_corrs))
    for i in range(len(canon_corrs)):
        z = (X_c @ U_cca[:, i]).reshape(-1, 1)
        beta = np.linalg.lstsq(z, sig_c, rcond=None)[0]
        ss_res = np.sum((sig_c - z @ beta) ** 2)
        per_r2[i] = max(1.0 - ss_res / ss_tot, 0.0)

    median_cc = np.median(canon_corrs)
    is_nuisance = (per_r2 <= epsilon) & (canon_corrs > median_cc)
    nu_hat = float(canon_corrs[is_nuisance].max()) if is_nuisance.any() else 0.0

    return kappa_hat, nu_hat


# ═══════════════════════════════════════════════════════════════════
# E7: Partial Recovery under Heterogeneous Signal Spectra (Prop 3.2)
# ═══════════════════════════════════════════════════════════════════
def experiment_e7():
    """
    E7: Partial Recovery under Heterogeneous Signal Spectra (Prop 3.2).

    With heterogeneous signal strengths κ_i = κ̄·ρ^{i-1}, the recovery
    count r (number of signal directions in the top-k recovered subspace)
    takes intermediate values 0 < r < k, smearing the sharp four-region
    phase diagram into a graded continuum.

    ρ=1 recovers the homogeneous case (sharp boundary, r ∈ {0, k}).
    A signal direction is "recovered" if its squared overlap with the
    span of the top-k recovered vectors exceeds 0.8.

    Figures:
      - E7_partial_recovery.png: 2×4 heatmap (CA/CP × ρ), color = r.
      - E7_theory_vs_empirical.png: scatter r_emp vs r_theory.
    """
    print("\n" + "=" * 60)
    print("E7: Partial Recovery (Prop 3.2)")
    print("=" * 60)

    d = D_DEFAULT  # 20
    k = 5          # k ≥ 5 so partial recovery has room to vary

    # Match phase_diagram.py panel (a): γ_x=1, γ_y=0.05, γ̃_y=5
    gamma_x_s_val = 1.0     # modality-x noise on signal coords
    gamma_y_s_val = 0.05    # modality-y noise on signal coords
    gamma_x_n_val = 1.0     # modality-x nuisance noise (homogeneous)
    gamma_y_n_val = 5.0     # modality-y nuisance noise (γ̃_y)

    gxs = np.ones(k) * gamma_x_s_val
    gys = np.ones(k) * gamma_y_s_val
    gxn = np.ones(d - k) * gamma_x_n_val
    gyn = np.ones(d - k) * gamma_y_n_val
    # ν = η / sqrt(γ̃_x · γ̃_y), matches theory.py and E4 convention
    max_eta = np.sqrt(gamma_x_n_val * gamma_y_n_val)

    OVERLAP_THRESH = 0.8
    rho_values = [1.0, 0.8, 0.6, 0.4]
    n_kbar = 31
    n_nu = 31
    kbar_vals = np.linspace(0, 3.0, n_kbar)
    nu_vals = np.linspace(0, 1.0, n_nu)

    # ── Compute theory and empirical recovery counts ──────────────
    all_data = []

    for rho_idx, rho in enumerate(rho_values):
        print(f"  ρ = {rho} ({rho_idx + 1}/{len(rho_values)})...")

        r_ca_th = np.zeros((n_nu, n_kbar))
        r_cp_th = np.zeros((n_nu, n_kbar))
        r_ca_emp = np.zeros((n_nu, n_kbar))
        r_cp_emp = np.zeros((n_nu, n_kbar))
        delta_ca_grid = np.zeros((n_nu, n_kbar))
        delta_cp_grid = np.zeros((n_nu, n_kbar))

        for i_k, kbar in enumerate(kbar_vals):
            kappas = kbar * rho ** np.arange(k)

            for i_n, nu in enumerate(nu_vals):
                etas = np.ones(d - k) * nu * max_eta

                # ── Theory (closed-form) ──
                preds = compute_recovery_predictions(
                    kappas, gxs, gys, gxn, gyn, etas)
                max_nu_th = np.max(preds.nu)
                max_xi_th = np.max(preds.xi)
                # r = |{i : signal_sv_i > max noise_sv_j}|  (Prop 3.2)
                r_ca_th[i_n, i_k] = np.sum(preds.rho > max_nu_th)
                r_cp_th[i_n, i_k] = np.sum(preds.tau > max_xi_th)
                delta_ca_grid[i_n, i_k] = min(preds.delta_ca, 100)
                delta_cp_grid[i_n, i_k] = min(preds.delta_cp, 100)

                # ── Empirical (finite-sample, average over seeds) ──
                ca_counts, cp_counts = [], []
                for seed in range(N_SEEDS):
                    data = generate_multimodal_data(
                        n=N_SAMPLES, d=d, k=k, kappas=kappas,
                        gamma_x_s=gxs, gamma_y_s=gys,
                        gamma_x_n=gxn, gamma_y_n=gyn,
                        etas=etas, seed=seed)

                    Sxx, Syy, Sxy = compute_sample_covariances(data.X, data.Y)
                    ca_sol = solve_ca(Sxx, Syy, Sxy, k)
                    cp_sol = solve_cp(Sxx, Syy, Sxy, k)

                    ca_sub = recovered_subspace(ca_sol.W, Sxx)
                    cp_sub = recovered_subspace(cp_sol.E, Sxx)

                    r_ca, r_cp = 0, 0
                    for i in range(k):
                        u_i = data.signal_subspace_x[:, i]
                        if np.sum((ca_sub.T @ u_i) ** 2) >= OVERLAP_THRESH:
                            r_ca += 1
                        if np.sum((cp_sub.T @ u_i) ** 2) >= OVERLAP_THRESH:
                            r_cp += 1
                    ca_counts.append(r_ca)
                    cp_counts.append(r_cp)

                r_ca_emp[i_n, i_k] = np.mean(ca_counts)
                r_cp_emp[i_n, i_k] = np.mean(cp_counts)

        all_data.append({
            'rho': rho,
            'r_ca_th': r_ca_th, 'r_cp_th': r_cp_th,
            'r_ca_emp': r_ca_emp, 'r_cp_emp': r_cp_emp,
            'delta_ca': delta_ca_grid, 'delta_cp': delta_cp_grid,
        })

    # ── Sanity check: ρ=1 theory must be sharp (only r=0 or r=k) ──
    rho1 = all_data[0]
    ca_th_unique = np.unique(rho1['r_ca_th'])
    cp_th_unique = np.unique(rho1['r_cp_th'])
    sharp = set(ca_th_unique).issubset({0, k}) and set(cp_th_unique).issubset({0, k})
    print(f"\n  Sanity check (ρ=1, theory):")
    print(f"    CA unique r: {ca_th_unique}")
    print(f"    CP unique r: {cp_th_unique}")
    print(f"    Sharp (only 0 and {k})? {'YES' if sharp else 'NO — investigate!'}")

    # ── Figure A: 2×4 heatmap grid ───────────────────────────────
    # Colors matching phase_diagram.py: beige (no recovery) → method color
    from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    C_NEITHER = '#F2F0EB'
    C_CA_ONLY = '#4A90C4'
    C_CP_ONLY = '#E07B54'
    C_CA_L = '#1A3A5C'
    C_CP_L = '#8B3A1E'

    def _make_cmap(c_from, c_to, n):
        """Sequential colormap from c_from to c_to with n levels."""
        r1, g1, b1 = to_rgb(c_from)
        r2, g2, b2 = to_rgb(c_to)
        colors = [(r1 + (r2 - r1) * i / (n - 1),
                   g1 + (g2 - g1) * i / (n - 1),
                   b1 + (b2 - b1) * i / (n - 1))
                  for i in range(n)]
        return ListedColormap(colors)

    cmap_ca = _make_cmap(C_NEITHER, C_CA_ONLY, k + 1)
    cmap_cp = _make_cmap(C_NEITHER, C_CP_ONLY, k + 1)
    bounds = np.arange(-0.5, k + 1.5, 1)
    norm = BoundaryNorm(bounds, k + 1)

    fig, axes = plt.subplots(2, 4, figsize=(24, 9),
                             constrained_layout=True)
    extent = [kbar_vals[0], kbar_vals[-1], nu_vals[0], nu_vals[-1]]

    for col, rd in enumerate(all_data):
        for row, (method, r_emp, cmap_row) in enumerate([
            ('CA', rd['r_ca_emp'], cmap_ca),
            ('CP', rd['r_cp_emp'], cmap_cp),
        ]):
            ax = axes[row, col]
            im = ax.imshow(r_emp, origin='lower', aspect='auto',
                           extent=extent, cmap=cmap_row, norm=norm,
                           interpolation='nearest')

            # Both Δ=1 contours on every panel (same as Fig 2)
            for delta, c_line, ls in [
                (rd['delta_ca'], C_CA_L, '-'),
                (rd['delta_cp'], C_CP_L, '--'),
            ]:
                if np.any(delta < 1) and np.any(delta > 1):
                    ax.contour(kbar_vals, nu_vals, delta,
                               levels=[1.0], colors=[c_line],
                               linewidths=[2.8], linestyles=[ls])

            ax.set_xlim(0, 3.0)
            ax.set_ylim(0, 1.0)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

            if row == 1:
                ax.set_xlabel(r'$\bar{\kappa}$', fontsize=20, fontweight='bold')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(r'$\nu$', fontsize=20, fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f'$\\rho = {rd["rho"]}$', fontsize=22,
                             fontweight='bold', pad=8)
            if col == 0:
                ax.text(-0.22, 0.5, method, transform=ax.transAxes,
                        fontsize=22, fontweight='bold', va='center',
                        ha='center', rotation=90)

            ax.tick_params(labelsize=16, direction='in')
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)

    # Separate colorbars for CA and CP rows
    cbar_ca = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_ca),
        ax=axes[0, :], location='right', shrink=0.8, pad=0.02,
        ticks=range(k + 1))
    cbar_ca.set_label('$r_{\\mathrm{CA}}$', fontsize=18, fontweight='bold')
    cbar_ca.ax.tick_params(labelsize=14)

    cbar_cp = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_cp),
        ax=axes[1, :], location='right', shrink=0.8, pad=0.02,
        ticks=range(k + 1))
    cbar_cp.set_label('$r_{\\mathrm{CP}}$', fontsize=18, fontweight='bold')
    cbar_cp.ax.tick_params(labelsize=14)

    # Legend for contour lines
    legend_elements = [
        Line2D([0], [0], color=C_CA_L, lw=2.8, ls='-',
               label='$\\Delta_{\\mathrm{CA}} = 1$'),
        Line2D([0], [0], color=C_CP_L, lw=2.8, ls='--',
               label='$\\Delta_{\\mathrm{CP}} = 1$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=14, frameon=True, edgecolor='#ccc',
               bbox_to_anchor=(0.45, -0.01),
               prop={'weight': 'bold'})

    fig.savefig(os.path.join(FIGURES_DIR, "E7_partial_recovery.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  → Saved E7_partial_recovery.png")

    # ── Figure B: r_empirical vs r_theory scatter ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors_rho = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22']
    for idx, rd in enumerate(all_data):
        rho = rd['rho']
        ax1.scatter(rd['r_ca_th'].ravel(), rd['r_ca_emp'].ravel(),
                    alpha=0.15, s=12, color=colors_rho[idx],
                    label=f'$\\rho={rho}$', rasterized=True)
        ax2.scatter(rd['r_cp_th'].ravel(), rd['r_cp_emp'].ravel(),
                    alpha=0.15, s=12, color=colors_rho[idx],
                    label=f'$\\rho={rho}$', rasterized=True)

    for ax, title in [(ax1, 'CA'), (ax2, 'CP')]:
        ax.plot([-0.5, k + 0.5], [-0.5, k + 0.5], 'k--', alpha=0.5,
                lw=1.5, label='$y = x$')
        ax.set_xlabel('$r_{\\mathrm{theory}}$', fontsize=18, fontweight='bold')
        ax.set_ylabel('$r_{\\mathrm{empirical}}$', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlim(-0.5, k + 0.5)
        ax.set_ylim(-0.5, k + 0.5)
        ax.set_xticks(range(k + 1))
        ax.set_yticks(range(k + 1))
        ax.legend(fontsize=11, loc='upper left')
        ax.tick_params(labelsize=14, direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E7_theory_vs_empirical.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  → Saved E7_theory_vs_empirical.png")


def experiment_e8():
    """
    E8: Estimation Validation — (κ̂, ν̂) vs true (κ, ν).

    Validates that the PLS/CCA-based estimation procedure recovers the
    true spiked covariance model parameters from finite-sample data.

    Three panels:
      (a) Sweep κ, compare estimated κ̂ (PLS SV) to true κ².
      (b) Sweep ν, compare estimated ν̂ (CCA) to true ν.
      (c) Phase diagram overlay — true vs estimated points.
    """
    print("\n" + "=" * 60)
    print("E8: Estimation Validation (κ̂, ν̂)")
    print("=" * 60)

    d = D_DEFAULT  # 20
    k = K_DEFAULT  # 3
    nc = d

    # Phase diagram parameters: large target noise regime
    gamma_s = 0.1
    gamma_n_val = 50.0
    gxs = np.ones(k) * gamma_s
    gys = np.ones(k) * gamma_s
    gxn = np.ones(d - k) * 1.0
    gyn = np.ones(d - k) * gamma_n_val
    max_eta = np.sqrt(gxn[0] * gyn[0])  # √50 ≈ 7.07

    def _sig(data):
        """Recover (n, k) shared signal s from Y_signal and Q_y."""
        return data.Y_signal @ data.Q_y[:, :k]

    # ── (a) Sweep κ ──────────────────────────────────────────────
    print("  (a) Sweeping κ...")
    kappa_vals = np.linspace(0.3, 4.0, 20)
    nu_fix = 0.3
    etas_fix = np.ones(d - k) * nu_fix * max_eta

    def run_kappa(kv, seed):
        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=np.ones(k) * kv,
            gamma_x_s=gxs, gamma_y_s=gys, gamma_x_n=gxn, gamma_y_n=gyn,
            etas=etas_fix, seed=seed)
        kh, _ = _estimate_kappa_nu(data.X, data.Y, _sig(data), nc)
        return {"kappa_hat": kh}

    res_k = _run_over_seeds(kappa_vals, run_kappa)

    # ── (b) Sweep ν ──────────────────────────────────────────────
    print("  (b) Sweeping ν...")
    nu_vals = np.linspace(0.05, 0.95, 20)
    kappa_fix = 2.0

    def run_nu(nv, seed):
        data = generate_multimodal_data(
            n=N_SAMPLES, d=d, k=k, kappas=np.ones(k) * kappa_fix,
            gamma_x_s=gxs, gamma_y_s=gys, gamma_x_n=gxn, gamma_y_n=gyn,
            etas=np.ones(d - k) * nv * max_eta, seed=seed)
        _, nh = _estimate_kappa_nu(data.X, data.Y, _sig(data), nc)
        return {"nu_hat": nh}

    res_n = _run_over_seeds(nu_vals, run_nu)

    # ── (c) Phase diagram grid ───────────────────────────────────
    print("  (c) Phase diagram grid...")
    k_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    n_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    pts = []
    for kv in k_grid:
        for nv in n_grid:
            kh_s, nh_s = [], []
            for seed in range(N_SEEDS):
                data = generate_multimodal_data(
                    n=N_SAMPLES, d=d, k=k, kappas=np.ones(k) * kv,
                    gamma_x_s=gxs, gamma_y_s=gys, gamma_x_n=gxn, gamma_y_n=gyn,
                    etas=np.ones(d - k) * nv * max_eta, seed=seed)
                kh, nh = _estimate_kappa_nu(data.X, data.Y, _sig(data), nc)
                kh_s.append(kh)
                nh_s.append(nh)
            pts.append((kv, nv, np.mean(kh_s), np.mean(nh_s)))

    # ── Plotting ─────────────────────────────────────────────────
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    C_NEITHER = '#F2F0EB'; C_CA = '#4A90C4'; C_CP = '#E07B54'; C_BOTH = '#7B6D8E'
    C_CA_L = '#1A3A5C'; C_CP_L = '#8B3A1E'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): κ̂ vs true κ²
    ax = axes[0]
    true_k2 = kappa_vals ** 2
    ax.plot(true_k2, res_k["kappa_hat_mean"], "o-", color="tab:blue", ms=5,
            label="Estimated $\\hat{\\kappa}$")
    ax.fill_between(true_k2,
                     res_k["kappa_hat_mean"] - res_k["kappa_hat_std"],
                     res_k["kappa_hat_mean"] + res_k["kappa_hat_std"],
                     alpha=0.2, color="tab:blue")
    lim_a = max(true_k2.max(), np.nanmax(res_k["kappa_hat_mean"])) * 1.05
    ax.plot([0, lim_a], [0, lim_a], "k--", alpha=0.4, label="$y = x$")
    ax.set_xlabel("True $\\kappa^2$ (population PLS singular value)")
    ax.set_ylabel("$\\hat{\\kappa}$ (estimated PLS SV)")
    ax.set_title("(a) Signal Strength Estimation")
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim_a)
    ax.set_ylim(0, lim_a)

    # Panel (b): ν̂ vs true ν
    ax = axes[1]
    ax.plot(nu_vals, res_n["nu_hat_mean"], "o-", color="tab:red", ms=5,
            label="Estimated $\\hat{\\nu}$")
    ax.fill_between(nu_vals,
                     res_n["nu_hat_mean"] - res_n["nu_hat_std"],
                     res_n["nu_hat_mean"] + res_n["nu_hat_std"],
                     alpha=0.2, color="tab:red")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="$y = x$")
    ax.set_xlabel("True $\\nu$ (nuisance correlation)")
    ax.set_ylabel("$\\hat{\\nu}$ (max nuisance CCA correlation)")
    ax.set_title("(b) Nuisance Correlation Estimation")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel (c): phase diagram with true and estimated points
    ax = axes[2]
    kmax = 5.0
    kline = np.linspace(0.001, kmax, 500)
    nline = np.linspace(0, 1.0, 500)
    K, N = np.meshgrid(kline, nline)

    ca_thresh = K ** 2 / (K ** 2 + gamma_s)
    cp_thresh = K ** 2 / (np.sqrt(K ** 2 + gamma_s) * np.sqrt(gamma_n_val))

    region = np.zeros_like(K, dtype=int)
    region[(N < ca_thresh) & ~(N < cp_thresh)] = 1   # CA only
    region[~(N < ca_thresh) & (N < cp_thresh)] = 2   # CP only
    region[(N < ca_thresh) & (N < cp_thresh)] = 3     # Both

    cmap = ListedColormap([C_NEITHER, C_CA, C_CP, C_BOTH])
    ax.imshow(region, origin='lower', aspect='auto',
              extent=[0, kmax, 0, 1.0], cmap=cmap, vmin=0, vmax=3,
              interpolation='nearest')

    nu_ca = kline ** 2 / (kline ** 2 + gamma_s)
    nu_cp = np.clip(kline ** 2 / (np.sqrt(kline ** 2 + gamma_s)
                                   * np.sqrt(gamma_n_val)), 0, 1)
    ax.plot(kline, nu_ca, color=C_CA_L, ls='-', lw=2, label='$\\Delta_{CA}=1$')
    ax.plot(kline, nu_cp, color=C_CP_L, ls='--', lw=2, label='$\\Delta_{CP}=1$')

    for kt, nt, kh_pls, nh in pts:
        ke = np.sqrt(max(kh_pls, 0))  # PLS SV ≈ κ² → √ for κ scale
        ax.plot(kt, nt, 'o', color='black', ms=5, zorder=10)
        ax.plot(ke, nh, '*', color='#2ECC71', ms=10, markeredgecolor='black',
                markeredgewidth=0.5, zorder=10)
        ax.annotate('', xy=(ke, nh), xytext=(kt, nt),
                     arrowprops=dict(arrowstyle='->', color='gray',
                                     lw=0.8, alpha=0.6))

    ax.plot([], [], 'ok', ms=5, label='True $(\\kappa, \\nu)$')
    ax.plot([], [], '*', color='#2ECC71', ms=10, markeredgecolor='black',
            markeredgewidth=0.5, label='Estimated $(\\hat{\\kappa}, \\hat{\\nu})$')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel("$\\kappa$ (signal strength)")
    ax.set_ylabel("$\\nu$ (nuisance correlation)")
    ax.set_title(f"(c) Phase Diagram ($\\gamma_s$={gamma_s}, $\\gamma_n$={int(gamma_n_val)})")
    ax.set_xlim(0, kmax)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "E7_estimation_validation.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → Saved E7_estimation_validation.png")


# ── Main ───────────────────────────────────────────────────────────
EXPERIMENTS = {
    "E1": ("Hard Bottleneck (Prop 2.1)", experiment_e1),
    "E1_multi": ("Hard Bottleneck — N Modalities", experiment_e1_multimodal),
    "E2": ("Soft Bottleneck (Prop 2.3)", experiment_e2),
    "E3": ("CA vs CP Separation (Cor 3.5)", experiment_e3),
    "E4": ("Noise Correlation (Props 3.1/3.3)", experiment_e4),
    "E5": ("CA+Probe vs CP (Prop 3.8)", experiment_e5),
    "E6": ("Joint Interpolation (Cor 3.12)", experiment_e6),
    "E7": ("Partial Recovery (Prop 3.2)", experiment_e7),
    "E8": ("Estimation Validation (κ̂, ν̂)", experiment_e8),
}


def main():
    parser = argparse.ArgumentParser(description="Run spiked synthetic experiments.")
    parser.add_argument("--exp", nargs="*", default=None, choices=list(EXPERIMENTS.keys()),
                        help="Experiments to run (default: all).")
    args = parser.parse_args()

    exps_to_run = args.exp or list(EXPERIMENTS.keys())

    print("Spiked Synthetic Experiment Suite")
    print(f"  Samples per run: {N_SAMPLES}")
    print(f"  Seeds per sweep point: {N_SEEDS}")
    print(f"  Experiments: {', '.join(exps_to_run)}")
    print(f"  Figures dir: {FIGURES_DIR}")

    for exp_id in exps_to_run:
        name, fn = EXPERIMENTS[exp_id]
        print(f"\n{'─' * 60}\nRunning {exp_id}: {name}\n{'─' * 60}")
        fn()

    print("\n" + "=" * 60)
    print("All experiments complete. Figures saved to:")
    print(f"  {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
