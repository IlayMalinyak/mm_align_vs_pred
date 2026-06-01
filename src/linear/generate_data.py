"""
Spiked covariance generative model for multimodal data.

Mirrors the block-diagonal signal/noise model from Section 3 of the paper.
Each modality is a sum of shared signal and modality-specific noise,
with explicit control over cross-modal noise correlation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SyntheticData:
    """Container for generated multimodal data and ground truth."""

    X: np.ndarray  # (n, d_x)
    Y: np.ndarray  # (n, d_y)
    Y_signal: np.ndarray  # (n, d_y) noiseless signal component of Y
    Q_x: np.ndarray  # (d_x, d_x) orthogonal basis for x
    Q_y: np.ndarray  # (d_y, d_y) orthogonal basis for y
    signal_subspace_x: np.ndarray  # (d_x, k) ground-truth signal directions in x
    signal_subspace_y: np.ndarray  # (d_y, k) ground-truth signal directions in y
    kappas: np.ndarray  # (k,) signal strengths
    S_xx: np.ndarray  # (d_x, d_x) population covariance
    S_yy: np.ndarray  # (d_y, d_y) population covariance
    S_xy: np.ndarray  # (d_x, d_y) population cross-covariance


def generate_multimodal_data(
    n: int,
    d: int,
    k: int,
    kappas: np.ndarray,
    gamma_x_s: np.ndarray,
    gamma_y_s: np.ndarray,
    gamma_x_n: np.ndarray,
    gamma_y_n: np.ndarray,
    etas: np.ndarray,
    Q_x: np.ndarray | None = None,
    Q_y: np.ndarray | None = None,
    seed: int | None = None,
) -> SyntheticData:
    """
    Generate multimodal data from the spiked covariance model.

    The model generates data in orthogonal bases Q_x, Q_y where the first k
    coordinates represent shared signal and the remaining d-k are nuisance.

    Covariance structure (in the bases Q_x, Q_y):
        S_xx = Q_x @ diag(κ² + γ_x_s, γ_x_n) @ Q_x.T
        S_yy = Q_y @ diag(κ² + γ_y_s, γ_y_n) @ Q_y.T
        S_xy = Q_x @ diag(κ²,         η)      @ Q_y.T

    Parameters
    ----------
    n : int
        Number of paired samples.
    d : int
        Ambient dimension (same for both modalities).
    k : int
        Signal dimension (number of shared factors).
    kappas : array (k,)
        Signal strengths, κ_1 >= ... >= κ_k > 0.
    gamma_x_s : array (k,)
        Modality-x noise variance on shared coordinates.
    gamma_y_s : array (k,)
        Modality-y noise variance on shared coordinates.
    gamma_x_n : array (d-k,)
        Modality-x noise variance on nuisance coordinates.
    gamma_y_n : array (d-k,)
        Modality-y noise variance on nuisance coordinates.
    etas : array (d-k,)
        Cross-modal noise correlation on nuisance coordinates.
        Must satisfy 0 <= eta_j <= sqrt(gamma_x_n_j * gamma_y_n_j).
    Q_x : array (d, d), optional
        Orthogonal basis for modality x. Random if None.
    Q_y : array (d, d), optional
        Orthogonal basis for modality y. Random if None.
    seed : int, optional
        Random seed.

    Returns
    -------
    SyntheticData
        Generated data with ground truth information.
    """
    rng = np.random.default_rng(seed)

    kappas = np.asarray(kappas, dtype=float)
    gamma_x_s = np.asarray(gamma_x_s, dtype=float)
    gamma_y_s = np.asarray(gamma_y_s, dtype=float)
    gamma_x_n = np.asarray(gamma_x_n, dtype=float)
    gamma_y_n = np.asarray(gamma_y_n, dtype=float)
    etas = np.asarray(etas, dtype=float)

    assert len(kappas) == k
    assert len(gamma_x_s) == k
    assert len(gamma_y_s) == k
    assert len(gamma_x_n) == d - k
    assert len(gamma_y_n) == d - k
    assert len(etas) == d - k
    assert np.all(etas >= 0)
    assert np.all(etas <= np.sqrt(gamma_x_n * gamma_y_n) + 1e-10)

    # Generate random orthogonal bases if not provided
    if Q_x is None:
        Q_x = _random_orthogonal(d, rng)
    if Q_y is None:
        Q_y = _random_orthogonal(d, rng)

    # --- Generate components in the canonical basis ---

    # 1. Shared signal: s ~ N(0, K²) where K = diag(kappas)
    signal = rng.standard_normal((n, k)) * kappas  # (n, k)

    # 2. Modality-specific noise on shared coords
    noise_x_shared = rng.standard_normal((n, k)) * np.sqrt(gamma_x_s)
    noise_y_shared = rng.standard_normal((n, k)) * np.sqrt(gamma_y_s)

    # 3. Nuisance noise with controlled cross-correlation
    #    Generate (n_x_n, n_y_n) jointly for each nuisance dimension j:
    #    Cov = [[gamma_x_n_j, eta_j], [eta_j, gamma_y_n_j]]
    noise_x_nuisance = np.zeros((n, d - k))
    noise_y_nuisance = np.zeros((n, d - k))

    for j in range(d - k):
        cov_j = np.array(
            [[gamma_x_n[j], etas[j]], [etas[j], gamma_y_n[j]]]
        )
        samples_j = rng.multivariate_normal(np.zeros(2), cov_j, size=n)
        noise_x_nuisance[:, j] = samples_j[:, 0]
        noise_y_nuisance[:, j] = samples_j[:, 1]

    # 4. Assemble in canonical basis then rotate
    x_canonical = np.concatenate(
        [signal + noise_x_shared, noise_x_nuisance], axis=1
    )  # (n, d)
    y_canonical = np.concatenate(
        [signal + noise_y_shared, noise_y_nuisance], axis=1
    )  # (n, d)

    X = x_canonical @ Q_x.T  # (n, d)
    Y = y_canonical @ Q_y.T  # (n, d)

    # --- Noiseless signal component of Y ---
    # signal is (n, k) in canonical basis; rotate to ambient y-space
    Y_signal = signal @ Q_y[:, :k].T  # (n, d)

    # --- Ground truth signal subspaces ---
    # The signal subspace in x-space is spanned by Q_x[:, :k]
    signal_subspace_x = Q_x[:, :k]  # (d, k), columns are signal directions
    signal_subspace_y = Q_y[:, :k]

    # --- Population covariance matrices ---
    S_xx_diag = np.concatenate([kappas**2 + gamma_x_s, gamma_x_n])
    S_yy_diag = np.concatenate([kappas**2 + gamma_y_s, gamma_y_n])
    S_xy_diag = np.concatenate([kappas**2, etas])

    S_xx = Q_x @ np.diag(S_xx_diag) @ Q_x.T
    S_yy = Q_y @ np.diag(S_yy_diag) @ Q_y.T
    S_xy = Q_x @ np.diag(S_xy_diag) @ Q_y.T

    return SyntheticData(
        X=X,
        Y=Y,
        Y_signal=Y_signal,
        Q_x=Q_x,
        Q_y=Q_y,
        signal_subspace_x=signal_subspace_x,
        signal_subspace_y=signal_subspace_y,
        kappas=kappas,
        S_xx=S_xx,
        S_yy=S_yy,
        S_xy=S_xy,
    )


def compute_sample_covariances(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample (cross-)covariance matrices."""
    n = X.shape[0]
    S_xx = (X.T @ X) / n
    S_yy = (Y.T @ Y) / n
    S_xy = (X.T @ Y) / n
    return S_xx, S_yy, S_xy


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    H = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(H)
    # Ensure positive diagonal of R for uniqueness
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q
