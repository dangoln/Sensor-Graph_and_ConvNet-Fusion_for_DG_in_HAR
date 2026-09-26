"""
P18b — feature-distribution alignment losses (source ↔ target-unlabeled).

These act in the LEARNED embedding space (the graph embedding h), pushing the
source and target feature distributions together so the classifier — trained on
source labels — transfers. No target labels are used.

  coral_loss : Deep CORAL (Sun & Saenko, 2016) — match 2nd-order statistics
               (feature covariances). Deterministic, cheap, stable.
  mmd_loss   : Maximum Mean Discrepancy with a multi-bandwidth Gaussian kernel
               (the DAN objective). Matches all moments via the kernel trick.

Both take Fs [Ns, d] (source features) and Ft [Nt, d] (target features) and return
a non-negative scalar tensor (0 = distributions identical under that criterion).
"""
import torch


def coral_loss(Fs: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared Frobenius distance between feature covariances,
    normalized by 4*d^2 (the paper's scaling). Needs >= 2 samples per side."""
    d = Fs.size(1)
    if Fs.size(0) < 2 or Ft.size(0) < 2:
        return Fs.new_zeros(())

    def _cov(F):
        Fc = F - F.mean(dim=0, keepdim=True)
        return (Fc.t() @ Fc) / (F.size(0) - 1)

    Cs, Ct = _cov(Fs), _cov(Ft)
    return (Cs - Ct).pow(2).sum() / (4.0 * d * d)


def _pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    a2 = (A * A).sum(dim=1, keepdim=True)        # [na,1]
    b2 = (B * B).sum(dim=1, keepdim=True).t()    # [1,nb]
    return (a2 + b2 - 2.0 * (A @ B.t())).clamp_min(0.0)


def mmd_loss(Fs: torch.Tensor, Ft: torch.Tensor,
             bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """Squared MMD with a sum of Gaussian RBF kernels. Bandwidths are scaled by
    the median pairwise distance of the pooled batch (the median heuristic)."""
    if Fs.size(0) < 2 or Ft.size(0) < 2:
        return Fs.new_zeros(())

    Z = torch.cat([Fs, Ft], dim=0)
    with torch.no_grad():
        med = _pairwise_sq_dists(Z, Z).median().clamp_min(1e-8)

    def _k(A, B):
        d2 = _pairwise_sq_dists(A, B)
        out = 0.0
        for bw in bandwidths:
            out = out + torch.exp(-d2 / (2.0 * bw * med))
        return out / len(bandwidths)

    return _k(Fs, Fs).mean() + _k(Ft, Ft).mean() - 2.0 * _k(Fs, Ft).mean()


def alignment_loss(method, Fs, Ft):
    """Dispatch by name. 'coral' | 'mmd' | 'none'/None -> 0."""
    if method == "coral":
        return coral_loss(Fs, Ft)
    if method == "mmd":
        return mmd_loss(Fs, Ft)
    return Fs.new_zeros(())
