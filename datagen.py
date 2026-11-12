"""
datagen.py — ε-greedy data generation (Algorithm 1 of the manuscript).

Runs ADMM instances; at each iteration, with probability eps the Z-update
uses the current (initially random) RNN proxy plus Gaussian exploration
noise, otherwise the exact batched-IPM solve; exact solves record
(features chi_{t,s}, label z_{s,0}) pairs. eps decays geometrically.

Streams pairs to an .npz shard per instance under the output directory.

Usage: python3 datagen.py OUT_DIR N_INST S T K_MAX SEED0 [RHO]
"""

import os
import sys
import time

import numpy as np
import torch

import data as datamod
import admm as adm
from rnn import ZProxy, RNNPredictor


def gen_instance(prob, net, K_max, rng, eps0=1.0, decay=0.02, floor=0.05,
                 sigma=0.5):
    """One ε-greedy trajectory; returns (feats, labels) lists."""
    pred = RNNPredictor(prob, net)
    S, G = prob.S, prob.G
    x = prob.x_lo + 0.5 * (prob.x_hi - prob.x_lo)
    lam = np.zeros((S, G))
    z0 = np.repeat(x[None, :], S, axis=0)
    feats, labels = [], []
    eps = eps0
    for k in range(1, K_max + 1):
        x = adm.x_update(prob, z0, lam)
        if rng.random() < eps:
            # exploration: proxy + Gaussian noise (no label available)
            zh, _, _ = pred(prob, x, lam)
            z0 = zh + sigma * rng.standard_normal(zh.shape)
            z0 = np.clip(z0, prob.x_lo[None, :], prob.x_hi[None, :])
        else:
            # exact step: record training pairs
            chi = pred.features(x, lam).numpy()          # (S,T,226)
            p = prob.z_solve(x, lam)
            z0 = p[:G].T
            feats.append(chi.astype(np.float32))
            labels.append(z0.astype(np.float32))
        lam = lam + prob.rho * (x[None, :] - z0)
        eps = max(floor, eps - decay)
    return feats, labels


def main():
    out_dir, n_inst, S, T, K_max, seed0 = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        int(sys.argv[5]), int(sys.argv[6]))
    rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
    os.makedirs(out_dir, exist_ok=True)
    case = datamod.load_case118()
    torch.manual_seed(seed0)
    net = ZProxy()   # random proxy for exploration
    for i in range(n_inst):
        t0 = time.time()
        scen = datamod.make_scenarios(case, S=S, T=T, seed=seed0 + i)
        prob = adm.Problem(case, scen, S=S, T=T, rho=rho)
        rng = np.random.default_rng(seed0 * 100003 + i)
        feats, labels = gen_instance(prob, net, K_max, rng)
        if feats:
            F = np.concatenate(feats, axis=0)
            L = np.concatenate(labels, axis=0)
            np.savez_compressed(os.path.join(out_dir, f"shard_{i:04d}.npz"),
                                feats=F, labels=L)
        print(f"inst {i}: {time.time()-t0:.0f}s pairs={sum(len(f) for f in feats)}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
