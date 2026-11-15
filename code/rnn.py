"""
rnn.py — RNN proxy for the ADMM Z-update and the Feasibility Restoration
Layer (FRL, Eq. (16) of the manuscript).

Architecture (manuscript Section 7.1): a stack of L=3 recurrent cells with
hidden width F_h=64, ReLU activations, spectral normalization on all weight
matrices, trained with Adam (lr 1e-3, batch 16).

Input at iteration k (per scenario s):  chi_{t,s} = { d_{t,s} (bus loads,
118), x^k (54), lambda_s^k (54) }  -> 226-dim per period; the network runs
over the T=24 periods and outputs zhat_{s,0} (54).

FRL: projection of zhat_{s,0} onto the set of points feasible for the
scenario's t0 elastic balance row, i.e.
    min_z  (1/2)||z - zhat||^2
    s.t.   x_lo <= z <= x_hi,   D_s0 - sh_max <= sum(z) <= D_s0 + cur_max
(a box + single range constraint), solved exactly by bisection on the
range-constraint multiplier.
"""

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- #
# RNN proxy                                                             #
# --------------------------------------------------------------------- #
def spectral_norm_linear(in_f, out_f, bias=True):
    lin = nn.Linear(in_f, out_f, bias=bias)
    return nn.utils.spectral_norm(lin)


class RecurrentCell(nn.Module):
    """h_t = ReLU(W_x x_t + W_h h_{t-1} + b), all maps spectral-normalized."""

    def __init__(self, in_dim, hid):
        super().__init__()
        self.Wx = spectral_norm_linear(in_dim, hid)
        self.Wh = spectral_norm_linear(hid, hid, bias=False)

    def forward(self, x, h):
        return torch.relu(self.Wx(x) + self.Wh(h))


class ZProxy(nn.Module):
    """L stacked recurrent cells over the time axis; readout of z_{s,0}."""

    def __init__(self, in_dim=226, hid=64, L=3, out_dim=54):
        super().__init__()
        self.L = L
        self.hid = hid
        dims = [in_dim] + [hid] * (L - 1)
        self.cells = nn.ModuleList(
            [RecurrentCell(dims[l], hid) for l in range(L)])
        self.readout = spectral_norm_linear(hid, out_dim)

    def forward(self, chi):
        """chi: (B, T, in_dim) -> (B, out_dim) prediction of z_{s,0}."""
        B, T, _ = chi.shape
        hs = [chi.new_zeros(B, self.hid) for _ in range(self.L)]
        for t in range(T):
            u = chi[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(u, hs[l])
                u = hs[l]
        return self.readout(hs[-1])


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


# --------------------------------------------------------------------- #
# FRL: box + range projection (Eq. 16 specialized)                      #
# --------------------------------------------------------------------- #
def frl_project(zhat, lo, hi, sum_lo, sum_hi, tol=1e-12, max_iter=100):
    """
    Exact Euclidean projection of zhat (S,G) onto
        { z : lo <= z <= hi,  sum_lo <= sum_g z_g <= sum_hi }   (row-wise).
    Bisection on the range multiplier; O(S*G*max_iter).
    Returns the projected array (S,G).
    """
    zhat = np.asarray(zhat, dtype=float)
    z = np.clip(zhat, lo[None, :], hi[None, :])
    total = z.sum(axis=1)
    upper = total > sum_hi + tol
    lower = total < sum_lo - tol
    # Fully batched bisection. Inactive rows remain at their box projection.
    a = np.zeros(zhat.shape[0])
    b = np.ones(zhat.shape[0])
    b[upper] = (zhat[upper] - lo).max(axis=1) + 1.0
    b[lower] = (hi - zhat[lower]).max(axis=1) + 1.0
    direction = np.zeros(zhat.shape[0])
    direction[upper] = -1.0
    direction[lower] = 1.0
    target = np.where(upper, sum_hi, sum_lo)
    active = upper | lower
    for _ in range(max_iter):
        nu = 0.5 * (a + b)
        trial = np.clip(zhat + direction[:, None] * nu[:, None], lo[None, :], hi[None, :])
        tot = trial.sum(axis=1)
        move_a = active & (((direction < 0) & (tot > target)) |
                           ((direction > 0) & (tot < target)))
        a[move_a] = nu[move_a]
        b[active & ~move_a] = nu[active & ~move_a]
    z[active] = np.clip(zhat[active] + direction[active, None] * b[active, None],
                        lo[None, :], hi[None, :])
    return z


# --------------------------------------------------------------------- #
# predict_fn adapter for admm.run_ml                                    #
# --------------------------------------------------------------------- #
class RNNPredictor:
    """Callable for admm.run_ml: RNN forward + FRL, timed separately."""

    def __init__(self, prob, net, device="cpu"):
        self.prob = prob
        self.net = net.to(device)
        self.device = device
        G, T, S = prob.G, prob.T, prob.S
        self.G, self.T, self.S = G, T, S
        # normalization stats (set by training; defaults harmless)
        self.feat_mean = np.zeros(226)
        self.feat_std = np.ones(226)
        self.lab_mean = np.zeros(54)
        self.lab_std = np.ones(54)

    def features(self, x, lam):
        """Build (S, T, 226) feature tensor from x (G,), lam (S,G)."""
        load = self.prob.load                      # (S,T,118)
        S, T, _ = load.shape
        chi = np.empty((S, T, 226), dtype=np.float32)
        chi[:, :, :118] = load
        chi[:, :, 118:172] = x[None, None, :]
        chi[:, :, 172:] = lam[:, None, :]
        chi = ((chi - self.feat_mean) / self.feat_std).astype(np.float32)
        return torch.from_numpy(chi).to(self.device)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "state_dict" not in ckpt:
            raise ValueError("Checkpoint must contain state_dict and normalization statistics")
        self.net.load_state_dict(ckpt["state_dict"], strict=True)
        for key in ("feat_mean", "feat_std", "lab_mean", "lab_std"):
            if key not in ckpt:
                raise ValueError(f"Checkpoint lacks normalization field: {key}")
            value = np.asarray(ckpt[key], dtype=float)
            setattr(self, key, value)
        self.net.eval()

    def __call__(self, prob, x, lam):
        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            chi = self.features(x, lam)
            zn = self.net(chi).cpu().numpy()       # (S,G), normalized space
        zhat = zn * self.lab_std + self.lab_mean
        t_fwd = time.perf_counter() - t0
        t1 = time.perf_counter()
        sum_lo = prob.D[0] - prob.D[0]             # sh_max = D[0] -> sum >= 0
        sum_hi = prob.D[0] + prob.W[0]             # sum <= D0 + W0
        z0 = frl_project(zhat, prob.x_lo, prob.x_hi, sum_lo, sum_hi)
        t_frl = time.perf_counter() - t1
        return z0, t_fwd, t_frl
