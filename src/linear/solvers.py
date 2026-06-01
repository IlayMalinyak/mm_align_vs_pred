"""
Closed-form solvers for multimodal learning objectives.

Implements:
- Cross-Alignment (CA) / CCA  —  Theorem 1.3
- Cross-Prediction (CP)       —  Proposition 1.6
- CA + Linear Probe            —  Eq. 3.7
- Joint Objective              —  Proposition 3.10
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CASolution:
    """Cross-alignment solution."""

    W: np.ndarray  # (k, d_x) encoder for x
    V: np.ndarray  # (k, d_y) encoder for y
    canonical_correlations: np.ndarray  # (k,) top-k canonical correlations


@dataclass
class CPSolution:
    """Cross-prediction solution."""

    B: np.ndarray  # (d_y, d_x) combined encoder-decoder
    E: np.ndarray  # (k, d_x) encoder part
    singular_values: np.ndarray  # (k,) top-k singular values of A


@dataclass
class JointSolution:
    """Joint objective solution."""

    W: np.ndarray  # (k, d_x) encoder for x
    V: np.ndarray  # (k, d_y) encoder for y via V* = W S_xy S_yy^-1
    B: np.ndarray  # (d_y, k) predictor
    eigenvalues: np.ndarray  # (k,) top-k eigenvalues of M


def _matrix_sqrt_inv(M: np.ndarray) -> np.ndarray:
    """Compute M^{-1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute M^{1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def solve_ca(
    S_xx: np.ndarray,
    S_yy: np.ndarray,
    S_xy: np.ndarray,
    k: int,
) -> CASolution:
    """
    Solve the Cross-Alignment (CA) / CCA problem (Theorem 1.3).

    Computes C = S_xx^{-1/2} S_xy S_yy^{-1/2} and takes the top-k SVD.

    Parameters
    ----------
    S_xx : array (d_x, d_x)
    S_yy : array (d_y, d_y)
    S_xy : array (d_x, d_y)
    k : int
        Shared dimension.

    Returns
    -------
    CASolution with W (k, d_x), V (k, d_y), canonical_correlations (k,).
    """
    S_xx_inv_sqrt = _matrix_sqrt_inv(S_xx)
    S_yy_inv_sqrt = _matrix_sqrt_inv(S_yy)

    C = S_xx_inv_sqrt @ S_xy @ S_yy_inv_sqrt
    P, Phi, Qt = np.linalg.svd(C, full_matrices=False)

    P_k = P[:, :k]
    Q_k = Qt[:k, :].T  # (d_y, k)
    Phi_k = Phi[:k]

    # W* = U Phi_k^{-1/2} P_k^T S_xx^{-1/2}  (choose U = I)
    Phi_k_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(Phi_k, 1e-12)))
    W = Phi_k_inv_sqrt @ P_k.T @ S_xx_inv_sqrt
    V = Phi_k_inv_sqrt @ Q_k.T @ S_yy_inv_sqrt

    return CASolution(W=W, V=V, canonical_correlations=Phi_k)


def solve_cp(
    S_xx: np.ndarray,
    S_yy: np.ndarray,
    S_xy: np.ndarray,
    k: int,
) -> CPSolution:
    """
    Solve the Cross-Prediction (CP) problem (Proposition 1.6).

    Computes A = S_yx S_xx^{-1/2} and takes its best rank-k approximation.

    Parameters
    ----------
    S_xx : array (d_x, d_x)
    S_yy : array (d_y, d_y)
    S_xy : array (d_x, d_y)
    k : int
        Rank constraint.

    Returns
    -------
    CPSolution with B (d_y, d_x), E (k, d_x), singular_values (k,).
    """
    S_xx_inv_sqrt = _matrix_sqrt_inv(S_xx)
    S_yx = S_xy.T

    A = S_yx @ S_xx_inv_sqrt  # (d_y, d_x)
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    U_k = U[:, :k]
    Sigma_k = Sigma[:k]
    V_k = Vt[:k, :]  # (k, d_x)

    # B* = U_k Sigma_k V_k^T S_xx^{-1/2}
    B = U_k @ np.diag(Sigma_k) @ V_k @ S_xx_inv_sqrt

    # The encoder part: E = V_k^T S_xx^{-1/2}
    E = V_k @ S_xx_inv_sqrt

    return CPSolution(B=B, E=E, singular_values=Sigma_k)


def solve_ca_probe(
    W_ca: np.ndarray,
    S_xx: np.ndarray,
    S_xy: np.ndarray,
) -> np.ndarray:
    """
    Solve the two-stage CA + linear probe (Eq. 3.7).

    Stage 1: use pre-computed CA encoder W_ca.
    Stage 2: fit B* = S_yx W^T (W S_xx W^T)^{-1}.

    Parameters
    ----------
    W_ca : array (k, d_x)
        CA encoder from solve_ca.
    S_xx : array (d_x, d_x)
    S_xy : array (d_x, d_y)

    Returns
    -------
    B : array (d_y, k)
        Linear probe predictor. Full prediction: B @ W_ca @ x.
    """
    S_yx = S_xy.T
    W_Sxx_Wt = W_ca @ S_xx @ W_ca.T  # (k, k)
    B = S_yx @ W_ca.T @ np.linalg.inv(W_Sxx_Wt)  # (d_y, k)
    return B


def solve_joint(
    S_xx: np.ndarray,
    S_yy: np.ndarray,
    S_xy: np.ndarray,
    k: int,
    lam: float,
) -> JointSolution:
    """
    Solve the joint objective (Proposition 3.10).

    min_{W,V,B}  E[||BWx - y||²] + λ E[||Wx - Vy||²]

    After profiling out B and V, the problem reduces to maximizing
    tr(W' M W'^T) where M = BB^T + λ(CC^T - I).

    Parameters
    ----------
    S_xx : array (d_x, d_x)
    S_yy : array (d_y, d_y)
    S_xy : array (d_x, d_y)
    k : int
        Shared dimension.
    lam : float
        Alignment weight λ >= 0.

    Returns
    -------
    JointSolution
    """
    S_xx_inv_sqrt = _matrix_sqrt_inv(S_xx)
    S_yy_inv_sqrt = _matrix_sqrt_inv(S_yy)

    # B_mat = S_xx^{-1/2} S_xy  (this is the B in the paper's corollary)
    B_mat = S_xx_inv_sqrt @ S_xy  # (d_x, d_y)
    C = S_xx_inv_sqrt @ S_xy @ S_yy_inv_sqrt  # canonical correlation matrix

    # M = B B^T + λ(C C^T - I)
    M = B_mat @ B_mat.T + lam * (C @ C.T - np.eye(S_xx.shape[0]))

    # Top-k eigenvectors of M
    eigvals, eigvecs = np.linalg.eigh(M)
    # eigh returns ascending order, take the last k
    idx = np.argsort(eigvals)[::-1][:k]
    W_prime = eigvecs[:, idx].T  # (k, d_x), rows are top-k eigenvectors
    top_eigenvalues = eigvals[idx]

    # W* = W' S_xx^{-1/2}
    W = W_prime @ S_xx_inv_sqrt

    # V* = W S_xy S_yy^{-1}
    S_yy_inv = np.linalg.inv(S_yy)
    V = W @ S_xy @ S_yy_inv

    # B* = S_yx W^T (W S_xx W^T)^{-1}
    S_yx = S_xy.T
    W_Sxx_Wt = W @ S_xx @ W.T
    B_pred = S_yx @ W.T @ np.linalg.inv(W_Sxx_Wt)

    return JointSolution(W=W, V=V, B=B_pred, eigenvalues=top_eigenvalues)
