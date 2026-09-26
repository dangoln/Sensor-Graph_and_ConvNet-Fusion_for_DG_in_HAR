"""
P18b — source↔target feature-alignment losses for the ConvNet (CORAL / MMD).

Same definitions as the GNN side (kept here too because the GNN and ConvNet run in
separate subprocesses with separate sys.path). Applied to the DeepConvLSTM's
last-timestep representation. No target labels used.
"""
import torch


def coral_loss(Fs: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
    d = Fs.size(1)
    if Fs.size(0) < 2 or Ft.size(0) < 2:
        return Fs.new_zeros(())

    def _cov(F):
        Fc = F - F.mean(dim=0, keepdim=True)
        return (Fc.t() @ Fc) / (F.size(0) - 1)

    Cs, Ct = _cov(Fs), _cov(Ft)
    return (Cs - Ct).pow(2).sum() / (4.0 * d * d)


def _pairwise_sq_dists(A, B):
    a2 = (A * A).sum(dim=1, keepdim=True)
    b2 = (B * B).sum(dim=1, keepdim=True).t()
    return (a2 + b2 - 2.0 * (A @ B.t())).clamp_min(0.0)


def mmd_loss(Fs, Ft, bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
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
    if method == "coral":
        return coral_loss(Fs, Ft)
    if method == "mmd":
        return mmd_loss(Fs, Ft)
    return Fs.new_zeros(())
