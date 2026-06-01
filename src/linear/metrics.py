"""
Evaluation metrics for multimodal recovery experiments.

Provides subspace distance, prediction error, and encoder norm metrics
matching the quantities analyzed in the paper.
"""

import numpy as np


def subspace_distance(U_true: np.ndarray, U_recovered: np.ndarray) -> float:
    """
    Compute subspace distance via principal angles.

    The distance is 1 - (1/k) sum(cos(theta_i)^2), where theta_i are
    the principal angles between the two subspaces. This is 0 for perfect
    recovery and 1 when subspaces are orthogonal.

    Parameters
    ----------
    U_true : array (d, k1)
        Columns span the true signal subspace.
    U_recovered : array (d, k2)
        Columns span the recovered subspace.

    Returns
    -------
    float
        Subspace distance in [0, 1].
    """
    # Orthonormalize both subspaces
    Q1, _ = np.linalg.qr(U_true)
    k1 = U_true.shape[1]
    Q1 = Q1[:, :k1]

    Q2, _ = np.linalg.qr(U_recovered)
    k2 = U_recovered.shape[1]
    Q2 = Q2[:, :k2]

    # Singular values of Q1^T Q2 are cos(principal angles)
    cos_angles = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    cos_angles = np.minimum(cos_angles, 1.0)  # numerical safety

    k = min(k1, k2)
    return 1.0 - np.sum(cos_angles[:k] ** 2) / k


def prediction_error(
    B: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    per_sample: bool = False,
) -> float | np.ndarray:
    """
    Compute mean squared prediction error.

    Parameters
    ----------
    B : array (d_y, d_x)
        Predictor matrix. Prediction = B @ x.
    X : array (n, d_x)
        Input samples.
    Y : array (n, d_y)
        Target samples.
    per_sample : bool
        If True, return per-sample errors.

    Returns
    -------
    float or array (n,)
        MSE(B @ X^T, Y^T) or per-sample errors.
    """
    residuals = Y - X @ B.T  # (n, d_y)
    sample_errors = np.sum(residuals**2, axis=1)  # (n,)
    if per_sample:
        return sample_errors
    return np.mean(sample_errors)


def encoder_norm(W: np.ndarray, S_xx: np.ndarray) -> float:
    """
    Compute encoder norm in the whitened space: ||W S_xx^{1/2}||_F^2.

    This is the ||W'||_F^2 quantity from the paper, where W' = W S_xx^{1/2}.

    Parameters
    ----------
    W : array (k, d_x)
        Encoder matrix.
    S_xx : array (d_x, d_x)
        Covariance matrix of modality x.

    Returns
    -------
    float
    """
    eigvals, eigvecs = np.linalg.eigh(S_xx)
    eigvals = np.maximum(eigvals, 1e-12)
    S_xx_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    W_prime = W @ S_xx_sqrt
    return np.sum(W_prime**2)


def encoder_norm_pair(
    W: np.ndarray,
    V: np.ndarray,
    S_xx: np.ndarray,
    S_yy: np.ndarray,
) -> float:
    """
    Compute combined encoder norm: ||W'||_F^2 + ||V'||_F^2.

    This is the quantity bounded by Proposition 2.3.
    """
    return encoder_norm(W, S_xx) + encoder_norm(V, S_yy)


def recovered_subspace(
    W: np.ndarray, S_xx: np.ndarray
) -> np.ndarray:
    """
    Extract the recovered subspace from an encoder W.

    The rows of W span a k-dimensional subspace of R^d.
    We return an orthonormal basis for this subspace as a (d, k) matrix.

    Parameters
    ----------
    W : array (k, d)
        Encoder matrix.
    S_xx : array (d, d)
        Covariance of modality x.

    Returns
    -------
    array (d, k)
        Columns span the recovered subspace (orthonormalized).
    """
    # SVD: W = U Σ V^T, where W is (k, d).
    # V^T has shape (k, d), so V has shape (d, k).
    # The columns of V span the row space of W in R^d.
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt.T  # (d, k)
