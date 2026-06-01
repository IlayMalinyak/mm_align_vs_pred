import torch
import torch.nn as nn


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(x, y):
    # Invariance
    sim_loss = nn.functional.mse_loss(x, y)

    # Variance
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.relu(1 - std_x)) + torch.mean(torch.relu(1 - std_y))

    # Covariance
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_y = (y.T @ y) / (y.size(0) - 1)
    cov_loss = (off_diagonal(cov_x).pow(2).sum() / x.size(1)) + \
               (off_diagonal(cov_y).pow(2).sum() / y.size(1))

    return 25.0 * sim_loss + 25.0 * std_loss + 1.0 * cov_loss


def deepcca_loss(H1, H2, outdim_size, use_all_singular_values=False):
    """Negative sum of top-k canonical correlations (DeepCCA objective).

    Adapted from Michaelvll/DeepCCA, updated for PyTorch 2.7:
    - torch.symeig replaced with torch.linalg.eigh
    - float64 for numerical stability
    """
    r1 = r2 = 1e-3
    eps = 1e-9

    H1, H2 = H1.double(), H2.double()
    o1 = o2 = H1.size(1)
    m = H1.size(0)

    H1bar = H1 - H1.mean(dim=0, keepdim=True)
    H2bar = H2 - H2.mean(dim=0, keepdim=True)

    SigmaHat12 = (1.0 / (m - 1)) * (H1bar.T @ H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * (H1bar.T @ H1bar) + r1 * torch.eye(o1, device=H1.device, dtype=torch.float64)
    SigmaHat22 = (1.0 / (m - 1)) * (H2bar.T @ H2bar) + r2 * torch.eye(o2, device=H1.device, dtype=torch.float64)

    # Root inverse via eigendecomposition (filter small eigenvalues)
    D1, V1 = torch.linalg.eigh(SigmaHat11)
    pos1 = D1 > eps
    S11_inv_sqrt = (V1[:, pos1] @ torch.diag(D1[pos1] ** -0.5)) @ V1[:, pos1].T

    D2, V2 = torch.linalg.eigh(SigmaHat22)
    pos2 = D2 > eps
    S22_inv_sqrt = (V2[:, pos2] @ torch.diag(D2[pos2] ** -0.5)) @ V2[:, pos2].T

    Tval = S11_inv_sqrt @ SigmaHat12 @ S22_inv_sqrt

    if use_all_singular_values:
        corr = torch.trace(torch.sqrt(Tval.T @ Tval))
    else:
        TT = Tval.T @ Tval
        TT = TT + r1 * torch.eye(TT.shape[0], device=TT.device, dtype=torch.float64)
        eigvals, _ = torch.linalg.eigh(TT)
        eigvals = torch.clamp(eigvals, min=eps)
        topk = eigvals.topk(outdim_size)[0]
        corr = torch.sum(torch.sqrt(topk))

    return (-corr).float()
