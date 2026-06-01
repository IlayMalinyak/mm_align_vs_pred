"""
Closed-form theoretical predictions from the paper.

Computes recovery thresholds, separation ratios, and soft bottleneck bounds
directly from the generative model parameters, without running any solver.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RecoveryPredictions:
    """Theoretical recovery quantities."""

    # CA singular values (Proposition 3.1)
    rho: np.ndarray  # (k,) signal singular values of C
    nu: np.ndarray  # (d-k,) noise singular values of C
    delta_ca: float  # CA separation ratio

    # CP singular values (Proposition 3.3)
    tau: np.ndarray  # (k,) signal singular values of A
    xi: np.ndarray  # (d-k,) noise singular values of A
    delta_cp: float  # CP separation ratio

    # Ratio
    delta_ratio: float  # Δ_CA / Δ_CP

    # Recovery flags
    ca_recovers: bool  # Δ_CA > 1
    cp_recovers: bool  # Δ_CP > 1


def compute_recovery_predictions(
    kappas: np.ndarray,
    gamma_x_s: np.ndarray,
    gamma_y_s: np.ndarray,
    gamma_x_n: np.ndarray,
    gamma_y_n: np.ndarray,
    etas: np.ndarray,
) -> RecoveryPredictions:
    """
    Compute all theoretical recovery predictions from model parameters.

    Parameters
    ----------
    kappas : array (k,)
        Signal strengths.
    gamma_x_s : array (k,)
        Modality-x noise on shared coords.
    gamma_y_s : array (k,)
        Modality-y noise on shared coords.
    gamma_x_n : array (d-k,)
        Modality-x noise on nuisance coords.
    gamma_y_n : array (d-k,)
        Modality-y noise on nuisance coords.
    etas : array (d-k,)
        Cross-modal noise correlation.

    Returns
    -------
    RecoveryPredictions
    """
    kappas = np.asarray(kappas, dtype=float)
    gamma_x_s = np.asarray(gamma_x_s, dtype=float)
    gamma_y_s = np.asarray(gamma_y_s, dtype=float)
    gamma_x_n = np.asarray(gamma_x_n, dtype=float)
    gamma_y_n = np.asarray(gamma_y_n, dtype=float)
    etas = np.asarray(etas, dtype=float)

    # CA singular values (Proposition 3.1)
    rho = kappas**2 / np.sqrt(
        (kappas**2 + gamma_x_s) * (kappas**2 + gamma_y_s)
    )
    nu = etas / np.sqrt(gamma_x_n * gamma_y_n + 1e-30)

    # CP singular values (Proposition 3.3)
    tau = kappas**2 / np.sqrt(kappas**2 + gamma_x_s)
    xi = etas / np.sqrt(gamma_x_n + 1e-30)

    # Separation ratios
    min_rho = np.min(rho) if len(rho) > 0 else np.inf
    max_nu = np.max(nu) if len(nu) > 0 else 0
    min_tau = np.min(tau) if len(tau) > 0 else np.inf
    max_xi = np.max(xi) if len(xi) > 0 else 0

    delta_ca = min_rho / max_nu if max_nu > 1e-15 else np.inf
    delta_cp = min_tau / max_xi if max_xi > 1e-15 else np.inf
    delta_ratio = delta_ca / delta_cp if delta_cp > 1e-15 else np.inf

    return RecoveryPredictions(
        rho=rho,
        nu=nu,
        delta_ca=delta_ca,
        tau=tau,
        xi=xi,
        delta_cp=delta_cp,
        delta_ratio=delta_ratio,
        ca_recovers=delta_ca > 1,
        cp_recovers=delta_cp > 1,
    )


def compute_ca_canonical_correlations(
    kappas: np.ndarray,
    gamma_x_s: np.ndarray,
    gamma_y_s: np.ndarray,
    gamma_x_n: np.ndarray,
    gamma_y_n: np.ndarray,
    etas: np.ndarray,
) -> np.ndarray:
    """
    Compute all singular values of C = S_xx^{-1/2} S_xy S_yy^{-1/2}.

    Returns the full sorted array (signal + noise singular values).
    """
    rho = kappas**2 / np.sqrt(
        (kappas**2 + gamma_x_s) * (kappas**2 + gamma_y_s)
    )
    nu = etas / np.sqrt(gamma_x_n * gamma_y_n + 1e-30)
    all_sv = np.concatenate([rho, nu])
    return np.sort(all_sv)[::-1]


def compute_soft_bottleneck_bound(phi_k: float) -> float:
    """
    Compute the soft bottleneck lower bound on encoder norms (Proposition 2.3).

    Any feasible CA solution satisfies ||W'||_F^2 + ||V'||_F^2 >= 2/phi_k.

    Parameters
    ----------
    phi_k : float
        The k-th canonical correlation (smallest selected).

    Returns
    -------
    float
        Lower bound: 2 / phi_k.
    """
    return 2.0 / max(phi_k, 1e-15)


def compute_separation_ratio_formula(
    kappas: np.ndarray,
    gamma_y_s: np.ndarray,
    gamma_y_n: np.ndarray,
) -> np.ndarray:
    """
    Compute the Δ_CA / Δ_CP ratio from Corollary 3.5.

    Δ_CA / Δ_CP = sqrt(max_j γ̃_y_j / min_i (κ_i² + γ_y_i^(s)))

    Parameters
    ----------
    kappas : array (k,)
    gamma_y_s : array (k,)
    gamma_y_n : array (d-k,)

    Returns
    -------
    float
        The ratio Δ_CA / Δ_CP.
    """
    max_gamma_y_n = np.max(gamma_y_n)
    min_signal_plus_noise = np.min(kappas**2 + gamma_y_s)
    return np.sqrt(max_gamma_y_n / min_signal_plus_noise)
