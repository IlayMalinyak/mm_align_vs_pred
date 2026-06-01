"""
Unified phase parameter estimation and phase diagram plotting.

Estimates all spiked covariance model parameters (kappa, nu, gamma_x, gamma_y,
gamma_tilde_x, gamma_tilde_y) from paired embeddings via SVD + PLS + CCA with
Ridge 5-fold CV for signal/nuisance classification.

Usage:
    python -m src.analyze_phase_diagram              # all experiments
    python -m src.analyze_phase_diagram imagenet      # single experiment
    python -m src.analyze_phase_diagram dsprites shapes3d  # multiple
    python -m src.analyze_phase_diagram --plot-only   # re-plot from cache
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import sys
import argparse
import pandas as pd
import pickle

# ── Quantile configuration (deprecated — kept for backward compat) ────
# New defaults: kappa=mean, nu=trimmed_max. Env vars still honored.
_KQ_RAW = os.environ.get('KAPPA_QUANTILE', 'mean')
_NQ_RAW = os.environ.get('NU_QUANTILE', 'trimmed_max')
KAPPA_QUANTILE = _KQ_RAW if _KQ_RAW in ('mean',) else float(_KQ_RAW)
NU_QUANTILE = _NQ_RAW if _NQ_RAW in ('mean', 'trimmed_max') else float(_NQ_RAW)


def quantile_suffix():
    """Return filename suffix based on aggregation settings."""
    if KAPPA_QUANTILE == 'mean' and NU_QUANTILE == 'trimmed_max':
        return ''  # new default — no suffix needed
    if KAPPA_QUANTILE == 'mean' and NU_QUANTILE == 'mean':
        return '_homogeneous'
    kq = int(KAPPA_QUANTILE) if isinstance(KAPPA_QUANTILE, float) and KAPPA_QUANTILE == int(KAPPA_QUANTILE) else KAPPA_QUANTILE
    nq = int(NU_QUANTILE) if isinstance(NU_QUANTILE, float) and NU_QUANTILE == int(NU_QUANTILE) else NU_QUANTILE
    return f"_q{kq}_{nq}"


def _aggregate_signal(values):
    """Aggregate signal components: always mean (kappa^2 = mean(s_pls[signal]))."""
    if KAPPA_QUANTILE == 'mean':
        return float(values.mean())
    return float(np.percentile(values, KAPPA_QUANTILE))


def _trimmed_max(values):
    """Trimmed max: drop largest, take second largest (for len>=3); else plain max."""
    if len(values) == 0:
        return 0.0
    if len(values) < 3:
        return float(np.max(values))
    sorted_vals = np.sort(values)[::-1]  # descending
    return float(sorted_vals[1])  # second largest


def _aggregate_nuisance(values):
    """Aggregate nuisance: trimmed_max (default), mean, or percentile."""
    if NU_QUANTILE == 'trimmed_max':
        return _trimmed_max(values)
    if NU_QUANTILE == 'mean':
        return float(values.mean())
    return float(np.percentile(values, NU_QUANTILE))

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.pad': 5,
    'ytick.major.pad': 5,
})

# Phase diagram colors (from phase_diagram.py)
C_NEITHER = '#F2F0EB'
C_CA_ONLY = '#4A90C4'
C_CP_ONLY = '#E07B54'
C_BOTH    = '#7B6D8E'
C_CA_LINE = '#1A3A5C'
C_CP_LINE = '#8B3A1E'

REGIME_COLORS = {
    'Both': C_BOTH, 'CA only': C_CA_ONLY,
    'CP only': C_CP_ONLY, 'Neither': '#888888',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Vectorized R² Computation
# ═══════════════════════════════════════════════════════════════════════

def _make_cv_folds(n, cv=5, seed=42):
    """Create CV fold index arrays (precompute once, reuse)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    sizes = np.full(cv, n // cv)
    sizes[:n % cv] += 1
    folds = []
    s = 0
    for fs in sizes:
        folds.append(idx[s:s + fs])
        s += fs
    return folds


def _r2_ridge_1d(score, labels, folds, alpha=1.0):
    """
    Vectorized R² of 1D score vs ALL label columns using Ridge CV.

    Parameters
    ----------
    score : (n,) array
    labels : (n, L) array
    folds : list of index arrays (from _make_cv_folds)

    Returns
    -------
    r2 : (L,) array — R² per label
    """
    L = labels.shape[1]
    ss_res = np.zeros(L)
    ss_tot = np.zeros(L)
    n = len(score)
    train_mask = np.ones(n, dtype=bool)

    for test_idx in folds:
        train_mask[:] = True
        train_mask[test_idx] = False

        x_tr = score[train_mask]
        y_tr = labels[train_mask]
        x_te = score[test_idx]
        y_te = labels[test_idx]

        # Center
        x_m = x_tr.mean()
        y_m = y_tr.mean(axis=0)
        x_c = x_tr - x_m

        # Ridge: w = x_c^T y_c / (x_c^T x_c + alpha)
        denom = np.dot(x_c, x_c) + alpha
        W = x_c @ (y_tr - y_m) / denom  # (L,)

        # Predict
        y_pred = np.outer(x_te - x_m, W) + y_m  # (n_te, L)

        ss_res += np.sum((y_te - y_pred) ** 2, axis=0)
        ss_tot += np.sum((y_te - y_te.mean(axis=0)) ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(ss_tot > 1e-15, 1.0 - ss_res / ss_tot, 0.0)
    return np.maximum(np.nan_to_num(r2, nan=0.0), 0.0)


def _r2_ridge_2d(X_score, labels, folds, alpha=1.0):
    """
    R² of 2D score matrix vs ALL label columns using Ridge CV.

    Parameters
    ----------
    X_score : (n, p) array — typically p=2 for PLS joint scores
    labels : (n, L) array

    Returns
    -------
    r2 : (L,) array
    """
    n, p = X_score.shape
    L = labels.shape[1]
    ss_res = np.zeros(L)
    ss_tot = np.zeros(L)
    train_mask = np.ones(n, dtype=bool)

    for test_idx in folds:
        train_mask[:] = True
        train_mask[test_idx] = False

        X_tr = X_score[train_mask]
        y_tr = labels[train_mask]
        X_te = X_score[test_idx]
        y_te = labels[test_idx]

        X_m = X_tr.mean(axis=0)
        y_m = y_tr.mean(axis=0)
        X_c = X_tr - X_m
        y_c = y_tr - y_m

        # Ridge: W = (X^T X + alpha I)^{-1} X^T Y
        A = X_c.T @ X_c + alpha * np.eye(p)
        W = np.linalg.solve(A, X_c.T @ y_c)  # (p, L)

        y_pred = (X_te - X_m) @ W + y_m
        ss_res += np.sum((y_te - y_pred) ** 2, axis=0)
        ss_tot += np.sum((y_te - y_te.mean(axis=0)) ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(ss_tot > 1e-15, 1.0 - ss_res / ss_tot, 0.0)
    return np.maximum(np.nan_to_num(r2, nan=0.0), 0.0)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Parameter Estimation Steps
# ═══════════════════════════════════════════════════════════════════════

def _classify_signal_elbow(r2_matrix, log_transform=True):
    """
    Classify components as signal or nuisance via piecewise-linear elbow detection
    on the sorted sum-R² curve (Section 3.5 of PARAMETER_ESTIMATION.md).

    Parameters
    ----------
    r2_matrix : (nc, L) array
    log_transform : bool — if True, fit elbow on log(sum_r2) to capture
        relative drop-offs; if False, fit on raw sum_r2.

    Algorithm:
      1. Compute sum_r2[i] = sum_j R²_{i,j}, clamp negatives to zero.
      2. Sort in decreasing order.
      3. Optionally transform to log-space.
      4. For each candidate breakpoint k in {1, ..., d-1}, fit two linear segments
         and compute total RSS.
      5. Select k* = argmin_k RSS(k).
      6. elbow_strength = max(1 - RSS(k*) / RSS_single, 0).

    Returns dict with:
        is_signal: (nc,) bool array in original component order
        n_signal: int
        sum_r2: (nc,) array
        elbow_strength: float in [0, 1]
        elbow_r2: float — R² at the transition point
    """
    sum_r2 = np.maximum(r2_matrix.sum(axis=1), 0.0)
    nc = len(sum_r2)

    if nc < 2:
        return {
            'is_signal': np.ones(nc, dtype=bool),
            'n_signal': nc,
            'sum_r2': sum_r2,
            'elbow_strength': 0.0,
            'elbow_r2': float(sum_r2[0]) if nc > 0 else 0.0,
        }

    # All R² <= 0: no signal
    if sum_r2.max() <= 0:
        return {
            'is_signal': np.zeros(nc, dtype=bool),
            'n_signal': 0,
            'sum_r2': sum_r2,
            'elbow_strength': 0.0,
            'elbow_r2': 0.0,
        }

    order = np.argsort(sum_r2)[::-1]  # descending
    sorted_r2 = sum_r2[order]
    x = np.arange(nc, dtype=np.float64)

    if log_transform:
        # Log-space: captures relative drop-offs rather than being
        # dominated by the largest values.  Floor prevents log(0).
        floor = max(sorted_r2.min() * 0.01, 1e-12)
        y = np.log(sorted_r2 + floor)
    else:
        y = sorted_r2.copy()

    # Single-segment RSS
    slope_single = np.polyfit(x, y, 1)
    pred_single = np.polyval(slope_single, x)
    rss_single = float(np.sum((y - pred_single) ** 2))

    # Two-segment RSS for each breakpoint
    best_k = 1
    best_rss = np.inf

    for k in range(1, nc):
        # Left segment: indices 0..k-1
        x_left = x[:k]
        y_left = y[:k]
        if len(x_left) == 1:
            rss_left = 0.0
        else:
            slope_l = np.polyfit(x_left, y_left, 1)
            rss_left = float(np.sum((y_left - np.polyval(slope_l, x_left)) ** 2))

        # Right segment: indices k..nc-1
        x_right = x[k:]
        y_right = y[k:]
        if len(x_right) == 1:
            rss_right = 0.0
        else:
            slope_r = np.polyfit(x_right, y_right, 1)
            rss_right = float(np.sum((y_right - np.polyval(slope_r, x_right)) ** 2))

        total_rss = rss_left + rss_right
        if total_rss < best_rss:
            best_rss = total_rss
            best_k = k

    elbow_strength = max(1.0 - best_rss / rss_single, 0.0) if rss_single > 1e-15 else 0.0
    elbow_r2 = float(sorted_r2[best_k - 1])  # last signal component

    # Map back to original order
    is_signal = np.zeros(nc, dtype=bool)
    is_signal[order[:best_k]] = True

    return {
        'is_signal': is_signal,
        'n_signal': best_k,
        'sum_r2': sum_r2,
        'elbow_strength': elbow_strength,
        'elbow_r2': elbow_r2,
    }


def _classify_signal_permutation(r2_matrix, epsilon_max, epsilon_sum):
    """
    Legacy permutation-based classification (kept for regression testing).

    Uses two criteria (OR):
      1. max per-label R² > epsilon_max  (strong single-label predictor)
      2. sum of per-label R² > epsilon_sum (weak multi-label predictor)
    """
    max_r2 = r2_matrix.max(axis=1)
    sum_r2 = r2_matrix.sum(axis=1)
    is_signal = (max_r2 > epsilon_max) | (sum_r2 > epsilon_sum)
    return {
        'is_signal': is_signal,
        'n_signal': int(is_signal.sum()),
        'sum_r2': sum_r2,
        'elbow_strength': float('nan'),
        'elbow_r2': float('nan'),
    }


def _variance_alignment_Q(singular_values, is_signal):
    """
    Compute variance-weighted alignment Q (Section 3.6 of PARAMETER_ESTIMATION.md).

    Q = sum_{i in signal} sigma_i^2 / sum_{i=1}^{k_signal} sigma_i^2

    where sigma_i are singular values in decreasing order.

    Q = 1 means signal-classified components are the top-variance directions.
    Q = 0 means no signal detected.
    """
    k_signal = int(is_signal.sum())
    if k_signal == 0:
        return 0.0

    var = singular_values ** 2
    numerator = float(var[is_signal].sum())
    # Top k_signal variance (the theoretical maximum)
    sorted_var = np.sort(var)[::-1]
    denominator = float(sorted_var[:k_signal].sum())

    if denominator < 1e-15:
        return 0.0
    return float(numerator / denominator)


def _permutation_threshold(scores_fn, labels, folds,
                           n_permutations=30, quantile=0.95, seed=12345):
    """
    Compute data-adaptive R² thresholds via label permutation.

    Shuffles labels n_permutations times and computes R² under the null
    (no association between component scores and labels). Returns the
    quantile-th percentile as thresholds for both max-R² and sum-R² criteria.

    Parameters
    ----------
    scores_fn : callable(labels_perm) -> r2_matrix (nc, L)
        Function that computes r2_matrix given (possibly permuted) labels.
    labels : (n, L) array
    folds : list of index arrays
    n_permutations : int
    quantile : float — percentile of null distribution (default 0.95)
    seed : int

    Returns
    -------
    epsilon_max : float — threshold for max-per-label R²
    epsilon_sum : float — threshold for sum-of-labels R²
    """
    rng = np.random.RandomState(seed)
    n = labels.shape[0]
    null_max, null_sum = [], []

    for _ in range(n_permutations):
        perm = rng.permutation(n)
        r2_null = scores_fn(labels[perm])
        null_max.append(r2_null.max(axis=1))   # (nc,)
        null_sum.append(r2_null.sum(axis=1))    # (nc,)

    # Pool across all components × permutations
    all_max = np.concatenate(null_max)
    all_sum = np.concatenate(null_sum)
    epsilon_max = float(np.percentile(all_max, quantile * 100))
    epsilon_sum = float(np.percentile(all_sum, quantile * 100))
    return epsilon_max, epsilon_sum


def _classify_all(r2_matrix, _compute_r2, labels, folds, n_permutations=30):
    """Run all three classifiers on the same R² matrix.

    Returns dict keyed by method name ('elbow_log', 'elbow', 'permutation').
    """
    results = {
        'elbow_log': _classify_signal_elbow(r2_matrix, log_transform=True),
        'elbow': _classify_signal_elbow(r2_matrix, log_transform=False),
    }
    eps_max, eps_sum = _permutation_threshold(
        _compute_r2, labels, folds, n_permutations=n_permutations)
    results['permutation'] = _classify_signal_permutation(r2_matrix, eps_max, eps_sum)
    return results


def _unimodal_svd(Z, labels, n_components, folds, n_permutations=30,
                   classification_method='elbow_log'):
    """Step 1: Unimodal SVD analysis for one modality."""
    n, d = Z.shape
    nc = min(n_components, d, n - 1)
    Z_c = Z - Z.mean(0)

    U, S, Vt = np.linalg.svd(Z_c, full_matrices=False)
    U = U[:, :nc]
    S = S[:nc]

    # Score vectors: U[:,i] * S[i]
    scores = U * S[np.newaxis, :]  # (n, nc)

    # R² per component per label
    def _compute_r2(lab):
        r2 = np.zeros((nc, lab.shape[1]))
        for i in range(nc):
            r2[i] = _r2_ridge_1d(scores[:, i], lab, folds)
        return r2

    r2_matrix = _compute_r2(labels)

    # Classify with all methods
    all_cls = _classify_all(r2_matrix, _compute_r2, labels, folds, n_permutations)

    # Primary classification (for backwards compat / plotting)
    classification = all_cls[classification_method]
    is_signal = classification['is_signal']

    # Variances = eigenvalues of sample covariance = S²/n
    variances = S ** 2 / n

    signal_var = float(variances[is_signal].mean()) if is_signal.any() else 0.0
    nuisance_var = float(variances[~is_signal].mean()) if (~is_signal).any() else 0.0

    # Variance alignment Q (only for unimodal SVD step)
    Q = _variance_alignment_Q(S, is_signal)

    # Per-method derived quantities
    methods = {}
    for mname, cls in all_cls.items():
        m_sig = cls['is_signal']
        m_sv = float(variances[m_sig].mean()) if m_sig.any() else 0.0
        m_nv = float(variances[~m_sig].mean()) if (~m_sig).any() else 0.0
        m_Q = _variance_alignment_Q(S, m_sig)
        methods[mname] = {
            'is_signal': m_sig, 'n_signal': cls['n_signal'],
            'elbow_strength': cls['elbow_strength'], 'elbow_r2': cls['elbow_r2'],
            'signal_var': m_sv, 'nuisance_var': m_nv, 'Q': m_Q,
        }

    return {
        'singular_values': S,
        'variances': variances,
        'r2_matrix': r2_matrix,
        'sum_r2': classification['sum_r2'],
        'is_signal': is_signal,
        'n_signal': classification['n_signal'],
        'elbow_strength': classification['elbow_strength'],
        'elbow_r2': classification['elbow_r2'],
        'signal_var': signal_var,
        'nuisance_var': nuisance_var,
        'Q': Q,
        'methods': methods,
    }


def _pls_analysis(Z1, Z2, labels, n_components, folds, n_permutations=30,
                   classification_method='elbow_log'):
    """Step 2: Cross-modal PLS for kappa estimation."""
    n = Z1.shape[0]
    nc = min(n_components, Z1.shape[1], Z2.shape[1])
    Z1_c = Z1 - Z1.mean(0)
    Z2_c = Z2 - Z2.mean(0)

    # Cross-covariance SVD
    C = Z1_c.T @ Z2_c / n
    U_pls, s_pls, Vt_pls = np.linalg.svd(C, full_matrices=False)
    V_pls = Vt_pls.T
    nc = min(nc, len(s_pls))
    U_pls = U_pls[:, :nc]
    V_pls = V_pls[:, :nc]
    s_pls = s_pls[:nc]

    # Precompute joint scores for all components
    joint_scores = []
    for i in range(nc):
        z1_score = Z1_c @ U_pls[:, i]
        z2_score = Z2_c @ V_pls[:, i]
        joint_scores.append(np.column_stack([z1_score, z2_score]))

    # R² per component using joint scores (z1_i, z2_i)
    def _compute_r2(lab):
        r2 = np.zeros((nc, lab.shape[1]))
        for i in range(nc):
            r2[i] = _r2_ridge_2d(joint_scores[i], lab, folds)
        return r2

    r2_matrix = _compute_r2(labels)

    # Classify with all methods
    all_cls = _classify_all(r2_matrix, _compute_r2, labels, folds, n_permutations)

    # Primary classification
    classification = all_cls[classification_method]
    is_signal = classification['is_signal']

    if is_signal.any():
        kappa_sq = _aggregate_signal(s_pls[is_signal])
        kappa_hat = np.sqrt(max(kappa_sq, 0.0))
    else:
        kappa_sq = 0.0
        kappa_hat = 0.0

    # Per-method derived quantities
    methods = {}
    for mname, cls in all_cls.items():
        m_sig = cls['is_signal']
        if m_sig.any():
            m_ksq = _aggregate_signal(s_pls[m_sig])
            m_khat = np.sqrt(max(m_ksq, 0.0))
        else:
            m_ksq, m_khat = 0.0, 0.0
        methods[mname] = {
            'is_signal': m_sig, 'n_signal': cls['n_signal'],
            'elbow_strength': cls['elbow_strength'], 'elbow_r2': cls['elbow_r2'],
            'kappa_sq': m_ksq, 'kappa_hat': m_khat,
        }

    return {
        'U_pls': U_pls,
        'V_pls': V_pls,
        'singular_values': s_pls,
        'r2_matrix': r2_matrix,
        'sum_r2': classification['sum_r2'],
        'is_signal': is_signal,
        'n_signal': classification['n_signal'],
        'elbow_strength': classification['elbow_strength'],
        'elbow_r2': classification['elbow_r2'],
        'kappa_sq': kappa_sq,
        'kappa_hat': kappa_hat,
        'methods': methods,
    }


def _cca_analysis(Z1, Z2, labels, n_components, folds, reg=1e-6,
                   n_permutations=30, classification_method='elbow_log'):
    """Step 3: CCA for nu estimation."""
    n = Z1.shape[0]
    nc = min(n_components, Z1.shape[1], Z2.shape[1])
    Z1_c = Z1 - Z1.mean(0)
    Z2_c = Z2 - Z2.mean(0)

    Sxx = Z1_c.T @ Z1_c / n + reg * np.eye(Z1.shape[1])
    Syy = Z2_c.T @ Z2_c / n + reg * np.eye(Z2.shape[1])
    Sxy = Z1_c.T @ Z2_c / n

    def _eigh_inv_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Sx_isqrt = _eigh_inv_sqrt(Sxx)
    Sy_isqrt = _eigh_inv_sqrt(Syy)
    M = Sx_isqrt @ Sxy @ Sy_isqrt
    P, phi, Qt = np.linalg.svd(M, full_matrices=False)

    nc = min(nc, len(phi))
    phi = np.clip(phi[:nc], 0, 1)
    W_cca = Sx_isqrt @ P[:, :nc]
    W_cca_y = Sy_isqrt @ Qt.T[:, :nc]

    # Precompute joint CCA scores (both modalities) for symmetric classification
    joint_scores = []
    for j in range(nc):
        z1_score = Z1_c @ W_cca[:, j]
        z2_score = Z2_c @ W_cca_y[:, j]
        joint_scores.append(np.column_stack([z1_score, z2_score]))

    # R² per CCA component using joint scores (symmetric w.r.t. modality swap)
    def _compute_r2(lab):
        r2 = np.zeros((nc, lab.shape[1]))
        for j in range(nc):
            r2[j] = _r2_ridge_2d(joint_scores[j], lab, folds)
        return r2

    r2_matrix = _compute_r2(labels)

    # Classify with all methods
    all_cls = _classify_all(r2_matrix, _compute_r2, labels, folds, n_permutations)

    # Primary classification
    classification = all_cls[classification_method]
    is_signal = classification['is_signal']
    is_nuisance = ~is_signal

    nu_hat = _aggregate_nuisance(phi[is_nuisance]) if is_nuisance.any() else 0.0

    # Per-method derived quantities
    methods = {}
    for mname, cls in all_cls.items():
        m_sig = cls['is_signal']
        m_nui = ~m_sig
        m_nu = _aggregate_nuisance(phi[m_nui]) if m_nui.any() else 0.0
        methods[mname] = {
            'is_signal': m_sig, 'n_signal': cls['n_signal'],
            'elbow_strength': cls['elbow_strength'], 'elbow_r2': cls['elbow_r2'],
            'nu_hat': m_nu,
        }

    return {
        'W_cca': W_cca,
        'canonical_correlations': phi,
        'r2_matrix': r2_matrix,
        'sum_r2': classification['sum_r2'],
        'is_signal': is_signal,
        'n_signal': classification['n_signal'],
        'elbow_strength': classification['elbow_strength'],
        'elbow_r2': classification['elbow_r2'],
        'nu_hat': nu_hat,
        'methods': methods,
    }


def _signal_set_overlap(is_signal_a, is_signal_b):
    """Compute |A∩B| / min(|A|, |B|) between two signal boolean masks."""
    a = np.asarray(is_signal_a, dtype=bool)
    b = np.asarray(is_signal_b, dtype=bool)
    na, nb = int(a.sum()), int(b.sum())
    if min(na, nb) == 0:
        return 0.0
    return float((a & b).sum()) / min(na, nb)


def _cp_direct_analysis(Z1, Z2, labels, n_components, folds, reg=1e-6,
                         n_permutations=30, classification_method='elbow'):
    """
    Step 3b: CP-direct — regression matrix A = Sigma_yx * Sigma_xx^{-1/2}.

    Computes SVD of A, classifies signal/nuisance on source-side scores,
    and returns Delta_CP_direct = min(sigma_A[signal]) / max(sigma_A[nuisance]).
    """
    n = Z1.shape[0]
    nc = min(n_components, Z1.shape[1], Z2.shape[1])
    Z1_c = Z1 - Z1.mean(0)
    Z2_c = Z2 - Z2.mean(0)

    Sxx = Z1_c.T @ Z1_c / n + reg * np.eye(Z1.shape[1])
    Syx = Z2_c.T @ Z1_c / n

    # Sigma_xx^{-1/2}
    def _eigh_inv_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Sx_isqrt = _eigh_inv_sqrt(Sxx)

    # A = Sigma_yx @ Sigma_xx^{-1/2}
    A = Syx @ Sx_isqrt
    U_a, sigma_a, Vt_a = np.linalg.svd(A, full_matrices=False)
    V_a = Vt_a.T

    nc = min(nc, len(sigma_a))
    sigma_a = sigma_a[:nc]
    V_a = V_a[:, :nc]

    # Source-side scores: Z1_c @ (Sigma_xx^{-1/2} @ V_a[:, j])
    W_source = Sx_isqrt @ V_a  # (d1, nc)

    def _compute_r2(lab):
        r2 = np.zeros((nc, lab.shape[1]))
        for j in range(nc):
            score_j = Z1_c @ W_source[:, j]
            r2[j] = _r2_ridge_1d(score_j, lab, folds)
        return r2

    r2_matrix = _compute_r2(labels)

    # Classify with all methods
    all_cls = _classify_all(r2_matrix, _compute_r2, labels, folds, n_permutations)

    # Primary classification
    classification = all_cls[classification_method]
    is_signal = classification['is_signal']
    is_nuisance = ~is_signal

    # Direct Delta_CP
    if is_signal.any() and is_nuisance.any():
        delta_cp_direct = float(sigma_a[is_signal].min() / max(sigma_a[is_nuisance].max(), 1e-15))
    elif is_signal.any():
        delta_cp_direct = float('inf')
    else:
        delta_cp_direct = 0.0

    # Per-method deltas
    methods = {}
    for mname, cls in all_cls.items():
        m_sig = cls['is_signal']
        m_nui = ~m_sig
        if m_sig.any() and m_nui.any():
            m_delta = float(sigma_a[m_sig].min() / max(sigma_a[m_nui].max(), 1e-15))
        elif m_sig.any():
            m_delta = float('inf')
        else:
            m_delta = 0.0
        methods[mname] = {
            'is_signal': m_sig, 'n_signal': cls['n_signal'],
            'elbow_strength': cls['elbow_strength'], 'elbow_r2': cls['elbow_r2'],
            'delta_cp_direct': m_delta,
        }

    return {
        'singular_values': sigma_a,
        'r2_matrix': r2_matrix,
        'sum_r2': classification['sum_r2'],
        'is_signal': is_signal,
        'n_signal': classification['n_signal'],
        'elbow_strength': classification['elbow_strength'],
        'elbow_r2': classification['elbow_r2'],
        'delta_cp_direct': delta_cp_direct,
        'methods': methods,
    }


def _compute_direct_deltas(cca_result, cp_direct_result):
    """Compute direct Delta_CA and Delta_CP from CCA and regression spectra."""
    phi = cca_result['canonical_correlations']
    is_signal_cca = cca_result['is_signal']
    is_nuisance_cca = ~is_signal_cca

    # Direct Delta_CA from CCA canonical correlations
    if is_signal_cca.any() and is_nuisance_cca.any():
        delta_ca_direct = float(phi[is_signal_cca].min() / max(phi[is_nuisance_cca].max(), 1e-15))
    elif is_signal_cca.any():
        delta_ca_direct = float('inf')
    else:
        delta_ca_direct = 0.0

    delta_cp_direct = cp_direct_result['delta_cp_direct']

    return delta_ca_direct, delta_cp_direct


def _classify_regime(delta_ca, delta_cp):
    """Classify regime from separation ratios."""
    if delta_ca > 1 and delta_cp > 1:
        return "Both"
    elif delta_ca > 1:
        return "CA only"
    elif delta_cp > 1:
        return "CP only"
    else:
        return "Neither"


def _derive_params(svd_x_m, svd_y_m, pls_m, cca_m,
                   delta_ca_direct=None, delta_cp_direct=None):
    """Derive full 6-parameter set + dual regime from per-method sub-results.

    Returns composite parameters (for visualization) and, if direct deltas
    are provided, the primary regime from direct-spectral estimation.
    """
    kappa_hat = pls_m['kappa_hat']
    kappa_sq = pls_m['kappa_sq']
    nu_hat = cca_m['nu_hat']

    gamma_x = max(svd_x_m['signal_var'] - kappa_sq, 0.0)
    gamma_y = max(svd_y_m['signal_var'] - kappa_sq, 0.0)
    gamma_tilde_x = svd_x_m['nuisance_var']
    gamma_tilde_y = svd_y_m['nuisance_var']

    # Composite separation ratios (secondary — for phase-plane visualization)
    if kappa_hat == 0:
        Delta_CA_comp, Delta_CP_comp = 0.0, 0.0
    elif nu_hat == 0:
        Delta_CA_comp, Delta_CP_comp = float('inf'), float('inf')
    else:
        denom_rho = np.sqrt(max((kappa_sq + gamma_x) * (kappa_sq + gamma_y), 1e-15))
        rho = kappa_sq / denom_rho
        tau = kappa_sq / np.sqrt(max(kappa_sq + gamma_x, 1e-15))
        xi = nu_hat * np.sqrt(max(gamma_tilde_y, 1e-15))
        Delta_CA_comp = rho / nu_hat
        Delta_CP_comp = tau / xi if xi > 0 else float('inf')

    regime_secondary = _classify_regime(Delta_CA_comp, Delta_CP_comp)

    # Direct regime (primary — authoritative)
    if delta_ca_direct is not None and delta_cp_direct is not None:
        regime_primary = _classify_regime(delta_ca_direct, delta_cp_direct)
    else:
        regime_primary = regime_secondary
        delta_ca_direct = Delta_CA_comp
        delta_cp_direct = Delta_CP_comp

    return {
        'kappa_hat': kappa_hat, 'nu_hat': nu_hat,
        'gamma_x_hat': gamma_x, 'gamma_y_hat': gamma_y,
        'gamma_tilde_x_hat': gamma_tilde_x, 'gamma_tilde_y_hat': gamma_tilde_y,
        # Composite (backward compat)
        'Delta_CA_hat': Delta_CA_comp, 'Delta_CP_hat': Delta_CP_comp,
        'delta_ca_composite': Delta_CA_comp, 'delta_cp_composite': Delta_CP_comp,
        'regime_secondary': regime_secondary,
        # Direct
        'delta_ca_direct': delta_ca_direct, 'delta_cp_direct': delta_cp_direct,
        'regime_primary': regime_primary,
        # Backward compat alias
        'predicted_regime': regime_primary,
        'regime_agreement': regime_primary == regime_secondary,
        'Q_x': svd_x_m['Q'], 'Q_y': svd_y_m['Q'],
    }


def estimate_parameters(Z1, Z2, labels, n_components=50, cv=5,
                        n_permutations=30,
                        classification_method='elbow'):
    """
    Full 6-parameter estimation from paired multimodal embeddings.

    Runs Steps 1-4 from PARAMETER_ESTIMATION.md:
      1. Unimodal SVD (both modalities)
      2. Cross-modal PLS (kappa)
      3a. CCA (nu, Delta_CA_direct)
      3b. CP-direct (Delta_CP_direct from regression matrix A)
      4. Derive composite parameters + dual regime prediction

    All three classifiers (elbow_log, elbow, permutation) run on the same
    decomposition. Returns results for each method alongside the primary.

    Parameters
    ----------
    Z1 : (n, d1) — first modality embeddings (source / modality x)
    Z2 : (n, d2) — second modality embeddings (target / modality y)
    labels : (n, L) — label matrix (one-hot or continuous)
    n_components : int — max SVD/PLS/CCA components to analyze
    cv : int — CV folds for Ridge R²
    n_permutations : int — number of label shuffles for adaptive threshold
    classification_method : str — primary method: 'elbow' (default),
        'elbow_log', or 'permutation'

    Returns
    -------
    dict with primary results (kappa_hat, nu_hat, ..., predicted_regime,
    regime_primary, regime_secondary, delta_ca_direct, delta_cp_direct,
    Q_x, Q_y, svd_x, svd_y, pls, cca, cp_direct) plus 'methods' sub-dict
    keyed by method name, each containing the full derivation.
    """
    folds = _make_cv_folds(Z1.shape[0], cv)

    print("    Step 1: Unimodal SVD ...", end=" ", flush=True)
    svd_x = _unimodal_svd(Z1, labels, n_components, folds, n_permutations,
                           classification_method)
    svd_y = _unimodal_svd(Z2, labels, n_components, folds, n_permutations,
                           classification_method)
    print(f"sig_x={svd_x['signal_var']:.4f} nui_x={svd_x['nuisance_var']:.4f}"
          f" Q_x={svd_x['Q']:.3f} (n_sig={svd_x['n_signal']},"
          f" elbow={svd_x['elbow_strength']:.2f})")

    print("    Step 2: PLS ...", end=" ", flush=True)
    pls = _pls_analysis(Z1, Z2, labels, n_components, folds, n_permutations,
                        classification_method)
    print(f"kappa={pls['kappa_hat']:.4f}"
          f" (n_sig={pls['n_signal']}, elbow={pls['elbow_strength']:.2f})")

    print("    Step 3a: CCA ...", end=" ", flush=True)
    cca = _cca_analysis(Z1, Z2, labels, n_components, folds,
                        n_permutations=n_permutations,
                        classification_method=classification_method)
    print(f"nu={cca['nu_hat']:.4f}"
          f" (n_sig={cca['n_signal']}, elbow={cca['elbow_strength']:.2f})")

    print("    Step 3b: CP-direct ...", end=" ", flush=True)
    cp_direct = _cp_direct_analysis(Z1, Z2, labels, n_components, folds,
                                     n_permutations=n_permutations,
                                     classification_method=classification_method)
    print(f"Delta_CP_direct={cp_direct['delta_cp_direct']:.4f}"
          f" (n_sig={cp_direct['n_signal']}, elbow={cp_direct['elbow_strength']:.2f})")

    # Step 4: Compute direct Deltas
    delta_ca_direct, delta_cp_direct = _compute_direct_deltas(cca, cp_direct)
    print(f"    => Direct: Delta_CA={delta_ca_direct:.4f}, "
          f"Delta_CP={delta_cp_direct:.4f} => "
          f"{_classify_regime(delta_ca_direct, delta_cp_direct)}")

    # Step 4: Derive primary parameters (composite + direct)
    primary = _derive_params(
        svd_x, svd_y,
        {'kappa_hat': pls['kappa_hat'], 'kappa_sq': pls['kappa_sq']},
        {'nu_hat': cca['nu_hat']},
        delta_ca_direct=delta_ca_direct,
        delta_cp_direct=delta_cp_direct,
    )

    print(f"    => kappa={primary['kappa_hat']:.4f}, nu={primary['nu_hat']:.4f}, "
          f"gx={primary['gamma_x_hat']:.4f}, gy={primary['gamma_y_hat']:.4f}, "
          f"gtx={primary['gamma_tilde_x_hat']:.4f}, gty={primary['gamma_tilde_y_hat']:.4f}")
    print(f"    => Composite: Delta_CA={primary['Delta_CA_hat']:.4f}, "
          f"Delta_CP={primary['Delta_CP_hat']:.4f} => {primary['regime_secondary']}")
    print(f"    => Primary regime: {primary['regime_primary']}"
          f" (agree={primary['regime_agreement']})")
    print(f"    => Q_x={primary['Q_x']:.3f}, Q_y={primary['Q_y']:.3f}")

    # Signal-set overlap diagnostics
    # Truncate to common length for overlap computation
    n_pls = len(pls['is_signal'])
    n_cca = len(cca['is_signal'])
    n_cpd = len(cp_direct['is_signal'])
    n_svd_x = len(svd_x['is_signal'])
    n_svd_y = len(svd_y['is_signal'])

    nc_min_pls_svd = min(n_pls, n_svd_x)
    overlap_svd_pls_x = _signal_set_overlap(
        svd_x['is_signal'][:nc_min_pls_svd], pls['is_signal'][:nc_min_pls_svd])
    nc_min_pls_svd_y = min(n_pls, n_svd_y)
    overlap_svd_pls_y = _signal_set_overlap(
        svd_y['is_signal'][:nc_min_pls_svd_y], pls['is_signal'][:nc_min_pls_svd_y])
    nc_min_pls_cca = min(n_pls, n_cca)
    overlap_pls_cca = _signal_set_overlap(
        pls['is_signal'][:nc_min_pls_cca], cca['is_signal'][:nc_min_pls_cca])

    print(f"    => Overlap: SVD_x∩PLS={overlap_svd_pls_x:.2f}, "
          f"SVD_y∩PLS={overlap_svd_pls_y:.2f}, PLS∩CCA={overlap_pls_cca:.2f}")

    # Step 5: Derive parameters for all methods
    method_names = ['elbow_log', 'elbow', 'permutation']
    methods = {}
    for mname in method_names:
        m_svd_x = svd_x['methods'][mname]
        m_svd_y = svd_y['methods'][mname]
        m_pls = pls['methods'][mname]
        m_cca = cca['methods'][mname]
        m_cpd = cp_direct['methods'][mname]

        # Per-method direct deltas
        m_phi = cca['canonical_correlations']
        m_is_sig_cca = m_cca['is_signal']
        m_is_nui_cca = ~m_is_sig_cca
        if m_is_sig_cca.any() and m_is_nui_cca.any():
            m_dca_direct = float(m_phi[m_is_sig_cca].min() / max(m_phi[m_is_nui_cca].max(), 1e-15))
        elif m_is_sig_cca.any():
            m_dca_direct = float('inf')
        else:
            m_dca_direct = 0.0
        m_dcp_direct = m_cpd['delta_cp_direct']

        methods[mname] = _derive_params(
            m_svd_x, m_svd_y, m_pls, m_cca,
            delta_ca_direct=m_dca_direct, delta_cp_direct=m_dcp_direct)

    result = dict(primary)
    result.update({
        'svd_x': svd_x,
        'svd_y': svd_y,
        'pls': pls,
        'cca': cca,
        'cp_direct': cp_direct,
        'methods': methods,
        'overlap_svd_pls_x': overlap_svd_pls_x,
        'overlap_svd_pls_y': overlap_svd_pls_y,
        'overlap_pls_cca': overlap_pls_cca,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: Plotting — Diagnostic
# ═══════════════════════════════════════════════════════════════════════

def _plot_svd_panel(ax, singular_values, r2_matrix, is_signal, title,
                    annotation, label_names=None, elbow_r2=None):
    """Plot one diagnostic panel (SVD, PLS, or CCA)."""
    nc = len(singular_values)
    x = np.arange(1, nc + 1)

    # Background coloring per component
    for i in range(nc):
        if is_signal[i]:
            color = '#d4edda'  # light green
        else:
            color = '#f8d7da'  # light red
        ax.axvspan(max(i + 0.5, 0.7), i + 1.5, alpha=0.4, color=color, zorder=0)

    # Sum R² curve (primary visualization for elbow)
    sum_r2 = np.maximum(r2_matrix.sum(axis=1), 0.0)
    ax.plot(x, sum_r2, 'k-', lw=1.8, label='sum $R^2$', zorder=4)

    # R² per label
    L = r2_matrix.shape[1]
    if L <= 10:
        colors = plt.cm.Set2(np.linspace(0, 1, max(L, 3)))
        for j in range(L):
            lbl = label_names[j] if label_names and j < len(label_names) else f"L{j}"
            ax.plot(x, r2_matrix[:, j], '-', color=colors[j], alpha=0.5,
                    lw=0.9, label=lbl, zorder=3)
    else:
        # Many labels: show range as faint band
        ax.fill_between(x, r2_matrix.min(axis=1), r2_matrix.max(axis=1),
                        alpha=0.15, color='steelblue', zorder=2)

    # Elbow threshold line (if available)
    if elbow_r2 is not None and np.isfinite(elbow_r2):
        ax.axhline(elbow_r2, color='black', ls='--', lw=0.8, alpha=0.6,
                   label=f'elbow $R^2$={elbow_r2:.3f}')

    ax.set_ylabel('$R^2$')
    ax.set_xlabel('Component index')
    ax.set_xscale('log')
    max_r2 = max(sum_r2.max(), 0.01)
    ax.set_ylim(-0.02, max_r2 * 1.15)

    # Singular values on twin axis
    ax2 = ax.twinx()
    ax2.bar(x, singular_values, alpha=0.25, color='gray', width=0.6, zorder=1)
    ax2.set_ylabel('$\\sigma_i$', color='gray')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#999', alpha=0.9))
    ax.set_xlim(0.7, nc + 0.5)
    ax.legend(fontsize=8, loc='center right', ncol=1)


def plot_diagnostic(result, name_a='View A', name_b='View B',
                    label_names=None, out_path=None, figsize=(24, 5.5)):
    """
    Plot 1 (_diagnostic): four-panel SVD/PLS/CCA/CP-direct diagnostic.

    Parameters
    ----------
    result : dict from estimate_parameters()
    name_a, name_b : str — modality names
    label_names : list of str — names for label columns
    out_path : str — save path (if None, show)
    """
    has_cp_direct = 'cp_direct' in result
    n_panels = 4 if has_cp_direct else 3
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    svd_x = result['svd_x']
    pls = result['pls']
    cca = result['cca']

    # Panel 1: Unimodal SVD of Z1
    ann = (f"signal_var={svd_x['signal_var']:.3f}\n"
           f"nuisance_var={svd_x['nuisance_var']:.3f}\n"
           f"#signal={svd_x['n_signal']}\n"
           f"Q={svd_x['Q']:.3f}\n"
           f"elbow={svd_x['elbow_strength']:.2f}")
    _plot_svd_panel(axes[0], svd_x['singular_values'], svd_x['r2_matrix'],
                    svd_x['is_signal'], f'{name_a} — Unimodal SVD',
                    ann, label_names, elbow_r2=svd_x.get('elbow_r2'))

    # Panel 2: PLS
    ann = (f"$\\hat{{\\kappa}}$={result['kappa_hat']:.4f}\n"
           f"$\\kappa^2$={pls['kappa_sq']:.4f}\n"
           f"#signal={pls['n_signal']}\n"
           f"elbow={pls['elbow_strength']:.2f}")
    _plot_svd_panel(axes[1], pls['singular_values'], pls['r2_matrix'],
                    pls['is_signal'], 'PLS ($\\kappa$)', ann, label_names,
                    elbow_r2=pls.get('elbow_r2'))

    # Panel 3: CCA
    dca = result.get('delta_ca_direct', result['Delta_CA_hat'])
    ann = (f"$\\hat{{\\nu}}$={result['nu_hat']:.4f}\n"
           f"$\\Delta_{{CA}}^{{dir}}$={dca:.3f}\n"
           f"#nuisance={(~cca['is_signal']).sum()}\n"
           f"elbow={cca['elbow_strength']:.2f}")
    _plot_svd_panel(axes[2], cca['canonical_correlations'], cca['r2_matrix'],
                    cca['is_signal'], 'CCA ($\\nu$, $\\Delta_{CA}$)', ann,
                    label_names, elbow_r2=cca.get('elbow_r2'))

    # Panel 4: CP-direct
    if has_cp_direct:
        cpd = result['cp_direct']
        dcp = cpd['delta_cp_direct']
        ann = (f"$\\Delta_{{CP}}^{{dir}}$={dcp:.3f}\n"
               f"#signal={cpd['n_signal']}\n"
               f"elbow={cpd['elbow_strength']:.2f}")
        _plot_svd_panel(axes[3], cpd['singular_values'], cpd['r2_matrix'],
                        cpd['is_signal'], 'CP-direct ($\\Delta_{CP}$)', ann,
                        label_names, elbow_r2=cpd.get('elbow_r2'))

    # Overall title
    dca_d = result.get('delta_ca_direct', result['Delta_CA_hat'])
    dcp_d = result.get('delta_cp_direct', result['Delta_CP_hat'])
    title = (f"{name_a} vs {name_b}  |  "
             f"$\\hat{{\\kappa}}$={result['kappa_hat']:.3f}, "
             f"$\\hat{{\\nu}}$={result['nu_hat']:.3f}  |  "
             f"Direct: $\\Delta_{{CA}}$={dca_d:.2f}, "
             f"$\\Delta_{{CP}}$={dcp_d:.2f} => "
             f"{result.get('regime_primary', result['predicted_regime'])}  |  "
             f"Composite: $\\Delta_{{CA}}$={result['Delta_CA_hat']:.2f}, "
             f"$\\Delta_{{CP}}$={result['Delta_CP_hat']:.2f} => "
             f"{result.get('regime_secondary', '')}  |  "
             f"$Q_x$={result.get('Q_x', 0):.2f}, "
             f"$Q_y$={result.get('Q_y', 0):.2f}")
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Plotting — Phase Diagram
# ═══════════════════════════════════════════════════════════════════════

def _ca_boundary(kappa, gamma_x, gamma_y):
    """nu at Delta_CA = 1."""
    k2 = kappa ** 2
    denom = np.sqrt(np.maximum((k2 + gamma_x) * (k2 + gamma_y), 1e-30))
    return k2 / denom


def _cp_boundary(kappa, gamma_x, gamma_tilde_y):
    """nu at Delta_CP = 1."""
    k2 = kappa ** 2
    denom = np.sqrt(np.maximum(k2 + gamma_x, 1e-30)) * np.sqrt(max(gamma_tilde_y, 1e-30))
    return k2 / denom


def plot_phase_diagram(markers, gamma_x, gamma_y, gamma_tilde_y,
                       kappa_max=3.0, out_path=None, title=None,
                       figsize=(8, 7), extra_handles=None,
                       data_legend_loc='lower left'):
    """
    Publication-quality phase diagram with estimated boundaries and data markers.

    Parameters
    ----------
    markers : list of dicts with keys:
        'kappa_hat', 'nu_hat'
        Optional: 'marker_shape' (default '*'), 'marker_size' (default 250),
                  'facecolor' (default 'white'), 'edgecolor' (default 'black')
    gamma_x, gamma_y, gamma_tilde_y : floats — boundary parameters
    kappa_max : float — x-axis limit
    out_path : str — save path
    title : str — figure title
    extra_handles : list — additional legend handles for data marker encoding
    data_legend_loc : str — location for data legend (default 'lower left')
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ── Shaded regions ──
    res = 800
    kappa = np.linspace(0, kappa_max, res)
    nu = np.linspace(0, 1.0, res)
    K, N = np.meshgrid(kappa, nu)

    ca_thresh = _ca_boundary(K, gamma_x, gamma_y)
    cp_thresh = np.clip(_cp_boundary(K, gamma_x, gamma_tilde_y), 0, 2.0)

    ca_ok = N < ca_thresh
    cp_ok = N < cp_thresh

    region = np.zeros_like(K, dtype=int)
    region[ca_ok & ~cp_ok] = 1  # CA only
    region[~ca_ok & cp_ok] = 2  # CP only
    region[ca_ok & cp_ok] = 3   # Both

    cmap = ListedColormap([C_NEITHER, C_CA_ONLY, C_CP_ONLY, C_BOTH])
    ax.imshow(region, origin='lower', aspect='auto',
              extent=[0, kappa_max, 0, 1.0], cmap=cmap, vmin=0, vmax=3,
              interpolation='bilinear')

    # ── Boundary curves ──
    kk = np.linspace(0.01, kappa_max, 800)
    nu_ca = _ca_boundary(kk, gamma_x, gamma_y)
    nu_cp = np.clip(_cp_boundary(kk, gamma_x, gamma_tilde_y), 0, 1.0)

    ax.plot(kk, nu_ca, color=C_CA_LINE, ls='-', lw=2.5, zorder=5)
    ax.plot(kk, nu_cp, color=C_CP_LINE, ls='--', lw=2.5, dashes=(6, 3),
            zorder=5)

    # ── Data markers with white halo ──
    for m in markers:
        kh = m['kappa_hat']
        nh = m['nu_hat']
        ms = m.get('marker_size', 250)
        shape = m.get('marker_shape', '*')
        fc = m.get('facecolor', 'white')
        ec = m.get('edgecolor', 'black')
        ax.scatter([kh], [nh], marker=shape, s=ms * 1.5,
                   c=['white'], edgecolors='white', linewidths=2.5, zorder=9)
        ax.scatter([kh], [nh], marker=shape, s=ms, c=[fc], edgecolors=ec,
                   linewidths=1.2, zorder=10)

    # ── Parameter text box ──
    txt = (f"$\\gamma_x = {gamma_x:.2f}$\n"
           f"$\\gamma_y = {gamma_y:.2f}$\n"
           f"$\\tilde{{\\gamma}}_y = {gamma_tilde_y:.2f}$")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=13,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#bbb', alpha=0.90, lw=0.6))

    # ── Data legend (marker encoding) ──
    if extra_handles:
        leg_data = ax.legend(handles=extra_handles, fontsize=12,
                             loc=data_legend_loc, framealpha=0.92,
                             edgecolor='#999', handletextpad=0.5)
        ax.add_artist(leg_data)

    # ── Region legend ──
    region_handles = [
        Patch(facecolor=C_NEITHER, edgecolor='#999', lw=0.8, label='Neither'),
        Patch(facecolor=C_CA_ONLY, edgecolor='#999', lw=0.8, label='CA only'),
        Patch(facecolor=C_CP_ONLY, edgecolor='#999', lw=0.8, label='CP only'),
        Patch(facecolor=C_BOTH, edgecolor='#999', lw=0.8, label='Both'),
        plt.Line2D([0], [0], color=C_CA_LINE, lw=2.5, ls='-',
                   label='$\\Delta_{\\mathrm{CA}}=1$'),
        plt.Line2D([0], [0], color=C_CP_LINE, lw=2.5, ls='--',
                   label='$\\Delta_{\\mathrm{CP}}=1$'),
    ]
    ax.legend(handles=region_handles, fontsize=12, loc='lower right',
              framealpha=0.92, edgecolor='#999')

    ax.set_xlabel(r'Signal strength $\hat{\kappa}$')
    ax.set_ylabel(r'Nuisance correlation $\hat{\nu}$')
    ax.set_xlim(0, kappa_max)
    ax.set_ylim(0, 1.0)

    if title:
        ax.set_title(title, pad=12)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4b: Combined Publication Phase Diagram (2×2)
# ═══════════════════════════════════════════════════════════════════════

def _plot_phase_panel(ax, markers, gamma_x, gamma_y, gamma_tilde_y,
                      kappa_max=3.0, extra_handles=None, panel_label=None,
                      title=None, show_gamma_box=True, show_ylabel=True,
                      data_legend_loc='lower left', data_legend_ncol=1):
    """Render one phase diagram panel (no legend — shared legend added externally)."""
    res = 1000
    kappa = np.linspace(0, kappa_max, res)
    nu = np.linspace(0, 1.0, res)
    K, N = np.meshgrid(kappa, nu)

    ca_thresh = _ca_boundary(K, gamma_x, gamma_y)
    cp_thresh = np.clip(_cp_boundary(K, gamma_x, gamma_tilde_y), 0, 2.0)

    ca_ok = N < ca_thresh
    cp_ok = N < cp_thresh

    region = np.zeros_like(K, dtype=int)
    region[ca_ok & ~cp_ok] = 1
    region[~ca_ok & cp_ok] = 2
    region[ca_ok & cp_ok] = 3

    cmap = ListedColormap([C_NEITHER, C_CA_ONLY, C_CP_ONLY, C_BOTH])
    ax.imshow(region, origin='lower', aspect='auto',
              extent=[0, kappa_max, 0, 1.0], cmap=cmap, vmin=0, vmax=3,
              interpolation='bilinear')

    # Boundary curves
    kk = np.linspace(0.01, kappa_max, 800)
    nu_ca = _ca_boundary(kk, gamma_x, gamma_y)
    nu_cp = np.clip(_cp_boundary(kk, gamma_x, gamma_tilde_y), 0, 1.0)

    ax.plot(kk, nu_ca, color=C_CA_LINE, ls='-', lw=2.0, zorder=5)
    ax.plot(kk, nu_cp, color=C_CP_LINE, ls='--', lw=2.0, dashes=(6, 3), zorder=5)

    # Data markers with white halo for visibility
    for m in markers:
        kh = m['kappa_hat']
        nh = m['nu_hat']
        ms = m.get('marker_size', 200)
        shape = m.get('marker_shape', '*')
        fc = m.get('facecolor', 'white')
        ec = m.get('edgecolor', 'black')
        # White halo
        ax.scatter([kh], [nh], marker=shape, s=ms * 1.5,
                   c=['white'], edgecolors='white', linewidths=2.5, zorder=9)
        # Actual marker
        ax.scatter([kh], [nh], marker=shape, s=ms, c=[fc], edgecolors=ec,
                   linewidths=1.2, zorder=10)

    # Parameter text box (compact)
    if show_gamma_box:
        txt = (f"$\\gamma_x\\!= {gamma_x:.2f}$\n"
               f"$\\gamma_y\\!= {gamma_y:.2f}$\n"
               f"$\\tilde{{\\gamma}}_y\\!= {gamma_tilde_y:.2f}$")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#bbb', alpha=0.88, lw=0.6))

    # Per-panel data legend (compact)
    if extra_handles:
        leg = ax.legend(handles=extra_handles, fontsize=8.5,
                        loc=data_legend_loc, framealpha=0.88, ncol=data_legend_ncol,
                        edgecolor='#bbb', handletextpad=0.3,
                        borderpad=0.35, labelspacing=0.3, columnspacing=0.8)
        leg.get_frame().set_linewidth(0.6)

    ax.set_xlabel(r'Signal strength $\hat{\kappa}$', fontsize=13, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel(r'Nuisance correlation $\hat{\nu}$', fontsize=13,
                       fontweight='bold')
    else:
        ax.set_ylabel('')
    ax.set_xlim(0, kappa_max)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=11)

    if panel_label:
        ax.text(-0.02, 1.04, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='right')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_combined_phase_diagrams(out_dir):
    """
    Create a 2×2 combined phase diagram from CSV data.

    Reads phase_estimates_{linear,dsprites,shapes3d,imagenet}.csv and produces
    a single publication-quality figure.
    """
    csv_dir = out_dir

    # ── Load CSVs ──
    panels = []

    # (a) Linear
    csv_path = os.path.join(csv_dir, 'phase_estimates_linear.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        valid = df[df['kappa_hat'] > 0]
        gx = valid['gamma_x'].median() if len(valid) else 0.5
        gy = valid['gamma_y'].median() if len(valid) else 0.5
        gty = valid['gamma_tilde_y'].median() if len(valid) else 2.0

        kappas_list = sorted(df['kappa_true'].unique())
        kappa_norm = plt.Normalize(vmin=min(kappas_list) - 0.2,
                                   vmax=max(kappas_list) + 0.2)
        kappa_cmap = plt.cm.viridis

        markers = []
        for _, row in df.iterrows():
            markers.append({
                'kappa_hat': row['kappa_hat'], 'nu_hat': row['nu_hat'],
                'facecolor': kappa_cmap(kappa_norm(row['kappa_true'])),
                'edgecolor': 'black', 'marker_shape': '*', 'marker_size': 200,
            })

        extra = [
            plt.Line2D([0], [0], marker='*', color='none',
                       markerfacecolor=kappa_cmap(kappa_norm(k)), markeredgecolor='black',
                       markersize=10, label=f'$\\kappa = {k}$')
            for k in kappas_list
        ]
        panels.append({'label': '(a)', 'title': 'Linear Model',
                       'markers': markers, 'gx': gx, 'gy': gy, 'gty': gty,
                       'kmax': 3.0, 'extra': extra,
                       'legend_loc': 'lower left', 'legend_ncol': 1})

    # (b) dSprites
    csv_path = os.path.join(csv_dir, 'phase_estimates_dsprites.csv')
    noise_colors = {0.2: '#3182BD', 0.5: '#FD8D3C', 0.9: '#DE2D26'}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        valid = df[df['kappa_hat'] > 0]
        gx = valid['gamma_x'].median() if len(valid) else 0.5
        gy = valid['gamma_y'].median() if len(valid) else 0.0
        gty = valid['gamma_tilde_y'].median() if len(valid) else 2.0

        markers = []
        for _, row in df.iterrows():
            markers.append({
                'kappa_hat': row['kappa_hat'], 'nu_hat': row['nu_hat'],
                'facecolor': noise_colors.get(row['noise'], '#999'),
                'edgecolor': 'black', 'marker_shape': '*', 'marker_size': 200,
            })

        extra = [
            plt.Line2D([0], [0], marker='*', color='none',
                       markerfacecolor=c, markeredgecolor='black',
                       markersize=10, label=f'$\\sigma_w = {n}$')
            for n, c in sorted(noise_colors.items())
        ]
        kmax = max(df['kappa_hat'].max() * 1.3 + 0.3, 1.5)
        panels.append({'label': '(b)', 'title': 'dSprites',
                       'markers': markers, 'gx': gx, 'gy': gy, 'gty': gty,
                       'kmax': kmax, 'extra': extra,
                       'legend_loc': 'lower left', 'legend_ncol': 1})

    # (c) Shapes3D
    csv_path = os.path.join(csv_dir, 'phase_estimates_shapes3d.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        valid = df[df['kappa_hat'] > 0]
        gx = valid['gamma_x'].median() if len(valid) else 0.5
        gy = valid['gamma_y'].median() if len(valid) else 0.0
        gty = valid['gamma_tilde_y'].median() if len(valid) else 10.0

        markers = []
        for _, row in df.iterrows():
            markers.append({
                'kappa_hat': row['kappa_hat'], 'nu_hat': row['nu_hat'],
                'facecolor': noise_colors.get(row['noise'], '#999'),
                'edgecolor': 'black', 'marker_shape': '*', 'marker_size': 200,
            })

        extra = [
            plt.Line2D([0], [0], marker='*', color='none',
                       markerfacecolor=c, markeredgecolor='black',
                       markersize=10, label=f'$\\sigma_w = {n}$')
            for n, c in sorted(noise_colors.items())
        ]
        kmax = max(df['kappa_hat'].max() * 1.3 + 0.3, 2.0)
        panels.append({'label': '(c)', 'title': 'Shapes3D',
                       'markers': markers, 'gx': gx, 'gy': gy, 'gty': gty,
                       'kmax': kmax, 'extra': extra,
                       'legend_loc': 'lower left', 'legend_ncol': 1})

    # (d) ImageNet-100
    csv_path = os.path.join(csv_dir, 'phase_estimates_imagenet.csv')
    noise_colors_imagenet = {0.0: '#3182BD', 0.2: '#FD8D3C', 0.5: '#DE2D26'}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Filter for pretrained encoder estimates if method column exists
        if 'method' in df.columns and (df['method'] == 'pretrained').any():
            df = df[df['method'] == 'pretrained']
        valid = df[df['kappa_hat'] > 0]
        gx = valid['gamma_x'].median() if len(valid) else 1.0
        gy = valid['gamma_y'].median() if len(valid) else 0.0
        gty = valid['gamma_tilde_y'].median() if len(valid) else 0.5

        markers = []
        for _, row in df.iterrows():
            markers.append({
                'kappa_hat': row['kappa_hat'], 'nu_hat': row['nu_hat'],
                'facecolor': noise_colors_imagenet.get(row['noise'], '#999'),
                'edgecolor': 'black',
                'marker_shape': 'D',
                'marker_size': 200,
            })

        extra = [
            plt.Line2D([0], [0], marker='D', color='none',
                       markerfacecolor=c, markeredgecolor='black',
                       markersize=10, label=f'$\\sigma_w = {n}$')
            for n, c in sorted(noise_colors_imagenet.items())
        ]
        kmax = min(max(df['kappa_hat'].max() * 1.3 + 0.3, 3.0), 8.0)
        panels.append({'label': '(d)', 'title': 'ImageNet-100',
                       'markers': markers, 'gx': gx, 'gy': gy, 'gty': gty,
                       'kmax': kmax, 'extra': extra,
                       'legend_loc': 'lower left', 'legend_ncol': 1})

    if not panels:
        print("  No CSV data found for combined plot.")
        return

    # ── Create figure ──
    n_panels = len(panels)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, p in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        _plot_phase_panel(ax, p['markers'], p['gx'], p['gy'], p['gty'],
                          kappa_max=p['kmax'], extra_handles=p['extra'],
                          panel_label=p['label'], title=p['title'],
                          show_ylabel=(col == 0),
                          data_legend_loc=p.get('legend_loc', 'lower left'),
                          data_legend_ncol=p.get('legend_ncol', 1))

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # ── Shared region legend at bottom ──
    region_handles = [
        Patch(facecolor=C_NEITHER, edgecolor='#999', lw=0.8, label='Neither'),
        Patch(facecolor=C_CA_ONLY, edgecolor='#999', lw=0.8, label='CA only'),
        Patch(facecolor=C_CP_ONLY, edgecolor='#999', lw=0.8, label='CP only'),
        Patch(facecolor=C_BOTH, edgecolor='#999', lw=0.8, label='Both'),
        plt.Line2D([0], [0], color=C_CA_LINE, lw=2.5, ls='-',
                   label='$\\Delta_{\\mathrm{CA}}=1$'),
        plt.Line2D([0], [0], color=C_CP_LINE, lw=2.5, ls='--',
                   label='$\\Delta_{\\mathrm{CP}}=1$'),
    ]
    fig.legend(handles=region_handles, loc='lower center', ncol=6,
               fontsize=11, frameon=True, edgecolor='#ccc', fancybox=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.0,
               handletextpad=0.5, columnspacing=1.8)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    fig.subplots_adjust(bottom=0.06)

    out_path = os.path.join(out_dir, 'phase_diagrams_combined.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.2)
    plt.close(fig)
    print(f"\n  Saved combined figure: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Result Caching
# ═══════════════════════════════════════════════════════════════════════

def _save_cache(all_results, markers, extra_meta, path):
    """Save full estimation results + markers + metadata to pickle cache."""
    cache = {
        'all_results': all_results,
        'markers': markers,
        'extra_meta': extra_meta,
    }
    with open(path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached: {path}")


def _load_cache(path):
    """Load cached results. Returns (all_results, markers, extra_meta)."""
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    print(f"  Loaded cache: {path}")
    return cache['all_results'], cache['markers'], cache['extra_meta']


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: Experiment Runners
# ═══════════════════════════════════════════════════════════════════════

def _make_onehot(labels, n_classes):
    """Convert integer labels to one-hot matrix."""
    n = len(labels)
    oh = np.zeros((n, n_classes))
    oh[np.arange(n), labels.astype(int)] = 1.0
    return oh


def _median_gammas(results):
    """
    Compute median gamma parameters for phase diagram boundaries.

    Only uses configs where kappa > 0 (signal detected).
    Degenerate configs (kappa=0) have gamma=0 by construction and would
    incorrectly drag the median to zero.
    """
    valid = [r for r in results if r['kappa_hat'] > 0]
    if not valid:
        valid = results  # fallback to all if none have signal
    gx = np.median([r['gamma_x_hat'] for r in valid])
    gy = np.median([r['gamma_y_hat'] for r in valid])
    gty = np.median([r['gamma_tilde_y_hat'] for r in valid])
    return gx, gy, gty


# ── Linear ───────────────────────────────────────────────────────────

def _plot_linear(all_results, markers, extra_meta, out_dir):
    """Plot phase diagram for the linear experiment from cached results."""
    kappas_list = extra_meta['kappas_list']
    gxs_val = extra_meta['gxs_val']
    gys_val = extra_meta['gys_val']
    gyn_val = extra_meta['gyn_val']

    kappa_norm = plt.Normalize(vmin=min(kappas_list) - 0.2,
                               vmax=max(kappas_list) + 0.2)
    kappa_cmap = plt.cm.viridis

    extra_handles = [
        plt.Line2D([0], [0], marker='*', color='none',
                   markerfacecolor=kappa_cmap(kappa_norm(k_val)),
                   markeredgecolor='black', markersize=14,
                   label=f'$\\kappa = {k_val}$')
        for k_val in kappas_list
    ]

    plot_phase_diagram(
        markers,
        gamma_x=gxs_val, gamma_y=gys_val, gamma_tilde_y=gyn_val,
        kappa_max=3.0,
        out_path=os.path.join(out_dir, 'phase_diagram_linear.png'),
        title='Linear Model',
        extra_handles=extra_handles,
    )

    rows = []
    for r in all_results:
        for mname, mres in r.get('methods', {}).items():
            rows.append({
                'kappa_true': r['kappa_true'], 'nu_true': r['nu_true'],
                'method': mname,
                'kappa_hat': mres['kappa_hat'], 'nu_hat': mres['nu_hat'],
                'gamma_x': mres['gamma_x_hat'], 'gamma_y': mres['gamma_y_hat'],
                'gamma_tilde_x': mres['gamma_tilde_x_hat'],
                'gamma_tilde_y': mres['gamma_tilde_y_hat'],
                'Delta_CA': mres['Delta_CA_hat'], 'Delta_CP': mres['Delta_CP_hat'],
                'regime': mres['predicted_regime'],
                'Q_x': mres.get('Q_x', float('nan')),
                'Q_y': mres.get('Q_y', float('nan')),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'phase_estimates_linear.csv'), index=False)
    print(f"\n  Saved CSV: {out_dir}/phase_estimates_linear.csv")


def run_linear(out_dir, plot_only=False):
    """Phase diagram for the linear/theoretical experiment."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'cache_linear.pkl')

    print("\n" + "=" * 60)
    print("Linear Experiment — Phase Estimation")
    print("=" * 60)

    if plot_only:
        all_results, markers, extra_meta = _load_cache(cache_path)
        _plot_linear(all_results, markers, extra_meta, out_dir)
        return

    from src.linear.generate_data import generate_multimodal_data

    d = 50
    k = 3
    n = 5000
    gxs_val = 0.5
    gys_val = 0.5
    gxn_val = 2.0
    gyn_val = 2.0

    kappas_list = [0.5, 1.0, 1.5, 2.0, 2.5]
    etas_list = [0.2, 0.5, 0.8, 1.0, 1.4, 1.8]

    all_results = []
    markers = []

    kappa_norm = plt.Normalize(vmin=min(kappas_list) - 0.2,
                               vmax=max(kappas_list) + 0.2)
    kappa_cmap = plt.cm.viridis

    for kappa_val in kappas_list:
        for eta_val in etas_list:
            nu_true = eta_val / np.sqrt(gxn_val * gyn_val)
            if nu_true > 1.0:
                continue

            kappas = np.full(k, kappa_val)
            etas = np.full(d - k, eta_val)

            print(f"\n  kappa={kappa_val}, eta={eta_val} (nu_true={nu_true:.3f})")
            data = generate_multimodal_data(
                n=n, d=d, k=k, kappas=kappas,
                gamma_x_s=np.full(k, gxs_val),
                gamma_y_s=np.full(k, gys_val),
                gamma_x_n=np.full(d - k, gxn_val),
                gamma_y_n=np.full(d - k, gyn_val),
                etas=etas, seed=42
            )

            signal = data.Y_signal
            labels = signal[:, :k]

            result = estimate_parameters(data.X, data.Y, labels,
                                         n_components=min(40, d))
            result['kappa_true'] = kappa_val
            result['nu_true'] = nu_true
            all_results.append(result)

            markers.append({
                'kappa_hat': result['kappa_hat'],
                'nu_hat': result['nu_hat'],
                'facecolor': kappa_cmap(kappa_norm(kappa_val)),
                'edgecolor': 'black',
                'marker_shape': '*',
                'marker_size': 280,
            })

    extra_meta = {'kappas_list': kappas_list, 'gxs_val': gxs_val,
                  'gys_val': gys_val, 'gyn_val': gyn_val}
    _save_cache(all_results, markers, extra_meta, cache_path)
    _plot_linear(all_results, markers, extra_meta, out_dir)


# ── dSprites ─────────────────────────────────────────────────────────

def _plot_dsprites(all_results, markers, extra_meta, out_dir):
    """Plot phase diagram for dSprites from cached results."""
    noise_colors = extra_meta['noise_colors']
    gx, gy, gty = _median_gammas(all_results)

    extra_handles = [
        plt.Line2D([0], [0], marker='*', color='none',
                   markerfacecolor=c, markeredgecolor='black',
                   markersize=14, label=f'$\\sigma_w = {n}$')
        for n, c in sorted(noise_colors.items())
    ]

    plot_phase_diagram(
        markers, gamma_x=gx, gamma_y=gy, gamma_tilde_y=gty,
        kappa_max=max(r['kappa_hat'] for r in all_results) * 1.3 + 0.5,
        out_path=os.path.join(out_dir, 'phase_diagram_dsprites.png'),
        title='dSprites',
        extra_handles=extra_handles,
    )

    rows = []
    for r in all_results:
        for mname, mres in r.get('methods', {}).items():
            rows.append({
                'jitter': r['jitter'], 'noise': r['noise'],
                'method': mname,
                'kappa_hat': mres['kappa_hat'], 'nu_hat': mres['nu_hat'],
                'gamma_x': mres['gamma_x_hat'], 'gamma_y': mres['gamma_y_hat'],
                'gamma_tilde_x': mres['gamma_tilde_x_hat'],
                'gamma_tilde_y': mres['gamma_tilde_y_hat'],
                'Delta_CA': mres['Delta_CA_hat'], 'Delta_CP': mres['Delta_CP_hat'],
                'regime': mres['predicted_regime'],
                'Q_x': mres.get('Q_x', float('nan')),
                'Q_y': mres.get('Q_y', float('nan')),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'phase_estimates_dsprites.csv'), index=False)


def run_dsprites(out_dir, plot_only=False):
    """Phase diagram for dSprites experiment."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'cache_dsprites.pkl')

    print("\n" + "=" * 60)
    print("dSprites Experiment — Phase Estimation")
    print("=" * 60)

    noise_colors = {0.2: '#3182BD', 0.5: '#FD8D3C', 0.9: '#DE2D26'}

    if plot_only:
        all_results, markers, extra_meta = _load_cache(cache_path)
        _plot_dsprites(all_results, markers, extra_meta, out_dir)
        return

    import gc
    from src.dsprites.data import StereoDSprites
    from torch.utils.data import DataLoader
    from sklearn.decomposition import PCA

    jitters = [0.0, 0.1, 0.2, 0.3, 0.5]
    noises = [0.2, 0.5, 0.9]
    n_samples = 3000
    pca_dim = 60
    n_components = 40

    all_results = []
    markers = []

    for jitter in jitters:
        for noise in noises:
            print(f"\n  jitter={jitter}, noise={noise}")

            ds = StereoDSprites(
                root="./src/dsprites", n_samples=n_samples,
                jitter_sigma=jitter, strong_noise_std=0.1,
                weak_noise_std=noise, pos_range=1.0, download=False,
            )
            loader = DataLoader(ds, batch_size=500, num_workers=0)
            X_list, Y_list, meta_list = [], [], []
            for x_b, y_b, m_b in loader:
                X_list.append(x_b.numpy().reshape(len(x_b), -1))
                Y_list.append(y_b.numpy().reshape(len(y_b), -1))
                meta_list.append(m_b.numpy())
            X = np.concatenate(X_list)
            Y = np.concatenate(Y_list)
            meta = np.concatenate(meta_list)
            labels_int = meta[:, 0].astype(int)
            labels = _make_onehot(labels_int, 3)

            X_pca = PCA(n_components=pca_dim).fit_transform(X)
            Y_pca = PCA(n_components=pca_dim).fit_transform(Y)

            result = estimate_parameters(X_pca, Y_pca, labels,
                                         n_components=n_components)
            result['jitter'] = jitter
            result['noise'] = noise
            all_results.append(result)

            markers.append({
                'kappa_hat': result['kappa_hat'],
                'nu_hat': result['nu_hat'],
                'facecolor': noise_colors[noise],
                'edgecolor': 'black',
                'marker_shape': '*',
                'marker_size': 280,
            })

            del X, Y, X_pca, Y_pca, ds, loader
            gc.collect()

    extra_meta = {'noise_colors': noise_colors}
    _save_cache(all_results, markers, extra_meta, cache_path)
    _plot_dsprites(all_results, markers, extra_meta, out_dir)


# ── Shapes3D ─────────────────────────────────────────────────────────

def _plot_shapes3d(all_results, markers, extra_meta, out_dir):
    """Plot phase diagram for Shapes3D from cached results."""
    noise_colors = extra_meta['noise_colors']
    gx, gy, gty = _median_gammas(all_results)

    extra_handles = [
        plt.Line2D([0], [0], marker='*', color='none',
                   markerfacecolor=c, markeredgecolor='black',
                   markersize=14, label=f'$\\sigma_w = {n}$')
        for n, c in sorted(noise_colors.items())
    ]

    plot_phase_diagram(
        markers, gamma_x=gx, gamma_y=gy, gamma_tilde_y=gty,
        kappa_max=max(r['kappa_hat'] for r in all_results) * 1.3 + 0.5,
        out_path=os.path.join(out_dir, 'phase_diagram_shapes3d.png'),
        title='Shapes3D',
        extra_handles=extra_handles,
    )

    rep_idx = len(all_results) // 2
    plot_diagnostic(
        all_results[rep_idx],
        name_a='View A (strong)', name_b='View B (weak)',
        label_names=['cube', 'cylinder', 'sphere', 'capsule'],
        out_path=os.path.join(out_dir, 'diagnostic_shapes3d.png'),
    )

    rows = []
    for r in all_results:
        for mname, mres in r.get('methods', {}).items():
            rows.append({
                'jitter': r['jitter'], 'noise': r['noise'],
                'method': mname,
                'kappa_hat': mres['kappa_hat'], 'nu_hat': mres['nu_hat'],
                'gamma_x': mres['gamma_x_hat'], 'gamma_y': mres['gamma_y_hat'],
                'gamma_tilde_x': mres['gamma_tilde_x_hat'],
                'gamma_tilde_y': mres['gamma_tilde_y_hat'],
                'Delta_CA': mres['Delta_CA_hat'], 'Delta_CP': mres['Delta_CP_hat'],
                'regime': mres['predicted_regime'],
                'Q_x': mres.get('Q_x', float('nan')),
                'Q_y': mres.get('Q_y', float('nan')),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'phase_estimates_shapes3d.csv'), index=False)


def run_shapes3d(out_dir, plot_only=False):
    """Phase diagram for Shapes3D experiment."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'cache_shapes3d.pkl')

    print("\n" + "=" * 60)
    print("Shapes3D Experiment — Phase Estimation")
    print("=" * 60)

    noise_colors = {0.2: '#3182BD', 0.5: '#FD8D3C', 0.9: '#DE2D26'}

    if plot_only:
        all_results, markers, extra_meta = _load_cache(cache_path)
        _plot_shapes3d(all_results, markers, extra_meta, out_dir)
        return

    from src.shapes3d.data import StereoShapes3D
    from torch.utils.data import DataLoader
    from sklearn.decomposition import PCA

    jitters = [0.0, 0.2, 0.5]
    noises = [0.2, 0.5, 0.9]
    n_samples = 10000
    pca_dim = 100
    n_components = 50

    all_results = []
    markers = []

    for jitter in jitters:
        for noise in noises:
            print(f"\n  jitter={jitter}, noise={noise}")

            ds = StereoShapes3D(
                root="./src/shapes3d", n_samples=n_samples,
                jitter_sigma=jitter, strong_noise_std=0.1,
                weak_noise_std=noise, pos_range=1.0, download=False,
            )
            loader = DataLoader(ds, batch_size=500, num_workers=0)
            X_list, Y_list, meta_list = [], [], []
            for x_b, y_b, m_b in loader:
                X_list.append(x_b.numpy().reshape(len(x_b), -1))
                Y_list.append(y_b.numpy().reshape(len(y_b), -1))
                meta_list.append(m_b.numpy())
            X = np.concatenate(X_list)
            Y = np.concatenate(Y_list)
            meta = np.concatenate(meta_list)
            labels_int = meta[:, 0].astype(int)
            labels = _make_onehot(labels_int, 4)

            X_pca = PCA(n_components=pca_dim).fit_transform(X)
            Y_pca = PCA(n_components=pca_dim).fit_transform(Y)

            result = estimate_parameters(X_pca, Y_pca, labels,
                                         n_components=n_components)
            result['jitter'] = jitter
            result['noise'] = noise
            all_results.append(result)

            markers.append({
                'kappa_hat': result['kappa_hat'],
                'nu_hat': result['nu_hat'],
                'facecolor': noise_colors[noise],
                'edgecolor': 'black',
                'marker_shape': '*',
                'marker_size': 280,
            })

    extra_meta = {'noise_colors': noise_colors}
    _save_cache(all_results, markers, extra_meta, cache_path)
    _plot_shapes3d(all_results, markers, extra_meta, out_dir)


# ── ImageNet ─────────────────────────────────────────────────────────

def _extract_pretrained_features(loader, device):
    """
    Extract 512-dim backbone features from frozen pretrained ResNet-18 encoders.

    Uses separate encoders for RGB (3-channel) and grayscale (1-channel) views.
    Returns (Z_A, Z_B, labels) as numpy arrays.
    """
    import torch
    from src.imagenet.models import ResNetEncoder

    # Frozen pretrained encoders — skip random projection head, use backbone only
    enc_a = ResNetEncoder(backbone='resnet18', in_channels=3, pretrained=True).to(device)
    enc_b = ResNetEncoder(backbone='resnet18', in_channels=1, pretrained=True).to(device)
    enc_a.eval()
    enc_b.eval()
    for p in enc_a.parameters():
        p.requires_grad_(False)
    for p in enc_b.parameters():
        p.requires_grad_(False)

    za_list, zb_list, label_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            xa = batch["view_A"].to(device)
            xb = batch["view_B"].to(device)
            # Use enc.net() to get 512-dim backbone features, skip random proj head
            za_list.append(enc_a.net(xa).cpu().numpy())
            zb_list.append(enc_b.net(xb).cpu().numpy())
            label_list.append(batch["label"].numpy())

    Z_A = np.concatenate(za_list)
    Z_B = np.concatenate(zb_list)
    labels = np.concatenate(label_list)

    # Free GPU memory
    del enc_a, enc_b
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return Z_A, Z_B, labels


def _extract_unimodal_ssl_features(loader, device, jitter, noise):
    """
    Extract 512-dim backbone features from unimodal VICReg-trained encoders.

    Loads trained encoders from checkpoints/imagenet_unimodal/unimodal_j{jitter}_n{noise}/best_model.pth.
    Returns (Z_A, Z_B, labels) as numpy arrays, or None if checkpoint not found.
    """
    import torch
    from src.imagenet.models import ResNetEncoder

    ckpt_dir = os.path.join(
        BASE_DIR, "checkpoints", "imagenet_unimodal",
        f"unimodal_j{jitter}_n{noise}")
    best_path = os.path.join(ckpt_dir, "best_model.pth")

    # Check for pre-extracted features first
    feat_dir = os.path.join(ckpt_dir, "features")
    if (os.path.exists(os.path.join(feat_dir, "Z_A.npy")) and
            os.path.exists(os.path.join(feat_dir, "Z_B.npy")) and
            os.path.exists(os.path.join(feat_dir, "labels.npy"))):
        print("      Loading pre-extracted features ...")
        Z_A = np.load(os.path.join(feat_dir, "Z_A.npy"))
        Z_B = np.load(os.path.join(feat_dir, "Z_B.npy"))
        labels = np.load(os.path.join(feat_dir, "labels.npy"))
        return Z_A, Z_B, labels

    if not os.path.exists(best_path):
        return None

    print(f"      Loading unimodal model from {best_path} ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    enc_a = ResNetEncoder(backbone='resnet18', output_dim=128,
                          in_channels=3, pretrained=False).to(device)
    enc_b = ResNetEncoder(backbone='resnet18', output_dim=128,
                          in_channels=1, pretrained=False).to(device)
    enc_a.load_state_dict(ckpt["enc_x"])
    enc_b.load_state_dict(ckpt["enc_y"])
    enc_a.eval()
    enc_b.eval()
    for p in enc_a.parameters():
        p.requires_grad_(False)
    for p in enc_b.parameters():
        p.requires_grad_(False)

    za_list, zb_list, label_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            xa = batch["view_A"].to(device)
            xb = batch["view_B"].to(device)
            # Use enc.net() for 512-dim backbone features, skip projection head
            za_list.append(enc_a.net(xa).cpu().numpy())
            zb_list.append(enc_b.net(xb).cpu().numpy())
            label_list.append(batch["label"].numpy())

    Z_A = np.concatenate(za_list)
    Z_B = np.concatenate(zb_list)
    labels = np.concatenate(label_list)

    del enc_a, enc_b
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return Z_A, Z_B, labels


def _plot_imagenet(all_results, markers, extra_meta, out_dir):
    """Plot phase diagram for ImageNet from cached results.

    Uses unimodal SSL estimates when available (best middle-ground representation),
    falls back to pretrained encoder estimates, then all results.
    """
    noise_colors = extra_meta['noise_colors']

    # Separate by method
    unimodal_results = [r for r in all_results if r.get('method') == 'unimodal_ssl']
    pretrained_results = [r for r in all_results if r.get('method') == 'pretrained']
    raw_results = [r for r in all_results if r.get('method') == 'raw_pixel']

    # Priority: unimodal_ssl > pretrained > all
    if unimodal_results:
        primary_results = unimodal_results
        primary_markers = [m for m in markers if m.get('method') == 'unimodal_ssl']
        primary_label = 'unimodal SSL'
    elif pretrained_results:
        primary_results = pretrained_results
        primary_markers = [m for m in markers if m.get('method') == 'pretrained']
        primary_label = 'pretrained encoder'
    else:
        primary_results = all_results
        primary_markers = markers
        primary_label = 'all methods'
    if not primary_markers:
        primary_markers = markers

    gx, gy, gty = _median_gammas(primary_results)

    extra_handles = [
        plt.Line2D([0], [0], marker='D', color='none',
                   markerfacecolor=c, markeredgecolor='black',
                   markersize=10, label=f'$\\sigma_w = {n}$')
        for n, c in sorted(noise_colors.items())
    ]

    kmax = max(r['kappa_hat'] for r in primary_results) * 1.3 + 0.5
    plot_phase_diagram(
        primary_markers, gamma_x=gx, gamma_y=gy, gamma_tilde_y=gty,
        kappa_max=min(kmax, 8.0),
        out_path=os.path.join(out_dir, 'phase_diagram_imagenet.png'),
        title=f'ImageNet-100 ({primary_label} features)',
        extra_handles=extra_handles,
    )

    # Diagnostic plot from pretrained results
    if primary_results:
        rep_idx = len(primary_results) // 2
        plot_diagnostic(
            primary_results[rep_idx],
            name_a='View A (RGB)', name_b='View B (grayscale)',
            out_path=os.path.join(out_dir, 'diagnostic_imagenet.png'),
        )

    # Save CSVs — pretrained as primary, raw pixel for reference
    def _results_to_rows(results):
        rows = []
        for r in results:
            rows.append({
                'jitter': r['jitter'], 'noise': r['noise'],
                'method': r.get('method', 'unknown'),
                'kappa_hat': r['kappa_hat'], 'nu_hat': r['nu_hat'],
                'gamma_x': r['gamma_x_hat'], 'gamma_y': r['gamma_y_hat'],
                'gamma_tilde_x': r['gamma_tilde_x_hat'],
                'gamma_tilde_y': r['gamma_tilde_y_hat'],
                'Delta_CA': r['Delta_CA_hat'], 'Delta_CP': r['Delta_CP_hat'],
                'regime': r['predicted_regime'],
            })
        return rows

    # Main CSV uses best available estimates (read by combined plot)
    if unimodal_results:
        pd.DataFrame(_results_to_rows(unimodal_results)).to_csv(
            os.path.join(out_dir, 'phase_estimates_imagenet.csv'), index=False)
    elif pretrained_results:
        pd.DataFrame(_results_to_rows(pretrained_results)).to_csv(
            os.path.join(out_dir, 'phase_estimates_imagenet.csv'), index=False)

    # Full CSV with both methods for comparison
    all_rows = _results_to_rows(all_results)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(out_dir, 'phase_estimates_imagenet_all.csv'), index=False)


def run_imagenet(out_dir, plot_only=False):
    """Phase diagram for ImageNet-100 with triple estimation:
    1. Unimodal VICReg SSL encoders (primary — best middle-ground representation)
    2. Pretrained ResNet-18 encoder features (reference)
    3. Raw pixels → downsample → PCA (reference)
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'cache_imagenet.pkl')

    print("\n" + "=" * 60)
    print("ImageNet-100 Experiment — Phase Estimation")
    print("  (unimodal SSL + pretrained encoder + raw pixels)")
    print("=" * 60)

    noise_colors = {0.0: '#3182BD', 0.2: '#FD8D3C', 0.5: '#DE2D26'}

    if plot_only:
        all_results, markers, extra_meta = _load_cache(cache_path)
        _plot_imagenet(all_results, markers, extra_meta, out_dir)
        return

    import gc
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.imagenet.data import StereoImageNet
    from sklearn.decomposition import PCA

    HF_CACHE = os.environ.get("HF_DATASETS_CACHE",
                               "${HF_HOME:-${HOME}/.cache/huggingface}/datasets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jitters = [0.0, 30.0, 50.0]
    noises = [0.0, 0.2, 0.5]
    n_samples = 5000
    target_size = 64   # downsample for raw pixel path
    pca_dim = 100
    n_components = 50

    print(f"  n_samples={n_samples}, device={device}")
    print(f"  Raw pixel path: downsample={target_size}x{target_size}, pca_dim={pca_dim}")
    print(f"  Pretrained path: ResNet-18 backbone (512-dim features)")

    all_results = []
    markers = []

    for jitter in jitters:
        for noise in noises:
            print(f"\n  jitter={jitter}, noise={noise}")

            # Load dataset at full 224x224 resolution (needed for encoder)
            ds = StereoImageNet(
                split="validation", cache_dir=HF_CACHE,
                jitter_sigma=jitter, weak_noise_std=noise,
                sigma_rot=10.0, max_samples=n_samples,
                seed_augmentation=True, augment=False,
            )
            loader = DataLoader(ds, batch_size=200, num_workers=0, shuffle=False)

            # ── Pretrained encoder path (primary) ──
            print("    [pretrained] Extracting ResNet-18 backbone features ...")
            Z_A, Z_B, labels_int = _extract_pretrained_features(loader, device)
            labels = _make_onehot(labels_int, 100)
            print(f"    [pretrained] Z_A: {Z_A.shape}, Z_B: {Z_B.shape}")

            result_pt = estimate_parameters(Z_A, Z_B, labels,
                                            n_components=n_components)
            result_pt['jitter'] = jitter
            result_pt['noise'] = noise
            result_pt['method'] = 'pretrained'
            all_results.append(result_pt)

            markers.append({
                'kappa_hat': result_pt['kappa_hat'],
                'nu_hat': result_pt['nu_hat'],
                'facecolor': noise_colors[noise],
                'edgecolor': 'black',
                'marker_shape': 'D',
                'marker_size': 200,
                'method': 'pretrained',
            })

            del Z_A, Z_B

            # ── Raw pixel path (reference) ──
            print("    [raw pixel] Downsampling + PCA ...")
            X_list, Y_list = [], []
            for batch in loader:
                xa = F.interpolate(batch["view_A"], size=target_size,
                                   mode='bilinear', align_corners=False)
                xb = F.interpolate(batch["view_B"], size=target_size,
                                   mode='bilinear', align_corners=False)
                X_list.append(xa.numpy().reshape(len(xa), -1))
                Y_list.append(xb.numpy().reshape(len(xb), -1))

            X = np.concatenate(X_list)
            Y = np.concatenate(Y_list)
            print(f"    [raw pixel] X: {X.shape}, Y: {Y.shape}")

            X_pca = PCA(n_components=pca_dim).fit_transform(X)
            Y_pca = PCA(n_components=pca_dim).fit_transform(Y)

            result_raw = estimate_parameters(X_pca, Y_pca, labels,
                                             n_components=n_components)
            result_raw['jitter'] = jitter
            result_raw['noise'] = noise
            result_raw['method'] = 'raw_pixel'
            all_results.append(result_raw)

            markers.append({
                'kappa_hat': result_raw['kappa_hat'],
                'nu_hat': result_raw['nu_hat'],
                'facecolor': noise_colors[noise],
                'edgecolor': 'black',
                'marker_shape': '*',
                'marker_size': 280,
                'method': 'raw_pixel',
            })

            del X, Y, X_pca, Y_pca

            # ── Unimodal SSL encoder path (primary when available) ──
            print("    [unimodal_ssl] Extracting VICReg-trained features ...")
            unimodal_feats = _extract_unimodal_ssl_features(
                loader, device, jitter, noise)
            if unimodal_feats is not None:
                Z_A_uni, Z_B_uni, labels_uni_int = unimodal_feats
                labels_uni = _make_onehot(labels_uni_int, 100)
                print(f"    [unimodal_ssl] Z_A: {Z_A_uni.shape}, "
                      f"Z_B: {Z_B_uni.shape}")

                result_uni = estimate_parameters(Z_A_uni, Z_B_uni, labels_uni,
                                                 n_components=n_components)
                result_uni['jitter'] = jitter
                result_uni['noise'] = noise
                result_uni['method'] = 'unimodal_ssl'
                all_results.append(result_uni)

                markers.append({
                    'kappa_hat': result_uni['kappa_hat'],
                    'nu_hat': result_uni['nu_hat'],
                    'facecolor': noise_colors[noise],
                    'edgecolor': 'black',
                    'marker_shape': 'o',
                    'marker_size': 200,
                    'method': 'unimodal_ssl',
                })

                del Z_A_uni, Z_B_uni
            else:
                print("    [unimodal_ssl] No trained model found — skipping")

            del ds, loader
            gc.collect()

    if not all_results:
        print("  No results — skipping plots")
        return

    extra_meta = {'noise_colors': noise_colors}
    _save_cache(all_results, markers, extra_meta, cache_path)
    _plot_imagenet(all_results, markers, extra_meta, out_dir)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: Main
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    'linear': run_linear,
    'dsprites': run_dsprites,
    'shapes3d': run_shapes3d,
    'imagenet': run_imagenet,
}


def main():
    parser = argparse.ArgumentParser(
        description='Phase parameter estimation and phase diagram plotting')
    parser.add_argument('experiments', nargs='*', default=['all'],
                        help='Experiments to run (linear, dsprites, shapes3d, imagenet, all)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Re-plot from cached results (skip estimation)')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined 2x2 publication figure from CSVs')
    args = parser.parse_args()

    out_dir = os.path.join(BASE_DIR, "src", "logs", "figures", "phase_diagrams")
    os.makedirs(out_dir, exist_ok=True)

    if args.combined:
        plot_combined_phase_diagrams(out_dir)
        print("\n" + "=" * 60)
        print(f"All outputs in: {out_dir}")
        print("=" * 60)
        return

    exps = args.experiments
    if 'all' in exps:
        exps = list(EXPERIMENTS.keys())

    for name in exps:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        try:
            EXPERIMENTS[name](out_dir, plot_only=args.plot_only)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Also generate combined figure if all CSVs exist
    csv_names = ['phase_estimates_linear.csv', 'phase_estimates_dsprites.csv',
                 'phase_estimates_shapes3d.csv', 'phase_estimates_imagenet.csv']
    if any(os.path.exists(os.path.join(out_dir, c)) for c in csv_names):
        plot_combined_phase_diagrams(out_dir)

    print("\n" + "=" * 60)
    print(f"All outputs in: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
