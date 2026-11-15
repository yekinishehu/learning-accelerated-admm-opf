"""
train.py — train the RNN proxy on the ε-greedy dataset (manuscript 7.1:
Adam lr 1e-3, batch 16, up to 50 epochs, early-stopping patience 5).

Usage: python3 train.py DATA_DIR OUT_CKPT [EPOCHS]
Saves OUT_CKPT (torch checkpoint incl. feature/label normalization stats)
and prints a small validation report.
"""

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from rnn import ZProxy


def load_dataset(data_dir):
    feats, labels = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "shard_*.npz"))):
        d = np.load(f)
        feats.append(d["feats"])
        labels.append(d["labels"])
    return np.concatenate(feats), np.concatenate(labels)


def main():
    data_dir, out_ckpt = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    F, L = load_dataset(data_dir)
    N = len(F)
    print(f"dataset: {N} pairs, feats {F.shape}, labels {L.shape}", flush=True)

    feat_mean = F.reshape(-1, F.shape[-1]).mean(axis=0)
    feat_std = F.reshape(-1, F.shape[-1]).std(axis=0) + 1e-6
    lab_mean = L.mean(axis=0)
    lab_std = L.std(axis=0) + 1e-6
    Fn = (F - feat_mean) / feat_std
    Ln = (L - lab_mean) / lab_std

    X = torch.from_numpy(Fn.astype(np.float32))
    Y = torch.from_numpy(Ln.astype(np.float32))
    n_val = max(1, int(0.1 * N))
    torch.manual_seed(0)
    np.random.seed(0)
    perm = torch.randperm(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    torch.manual_seed(0)
    net = ZProxy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best_val, bad = np.inf, 0
    batch = 16
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        p = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            idx = p[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val = lossf(net(Xva), Yva).item()
        print(f"epoch {ep}: train {tot/len(Xtr):.5f} val {val:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            torch.save({"state_dict": net.state_dict(),
                        "feat_mean": feat_mean, "feat_std": feat_std,
                        "lab_mean": lab_mean, "lab_std": lab_std}, out_ckpt)
        else:
            bad += 1
            if bad >= 5:
                print("early stop", flush=True)
                break
    # validation in physical units
    ck = torch.load(out_ckpt, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        pred = net(Xva).numpy() * lab_std + lab_mean
        true = Yva.numpy() * lab_std + lab_mean
    err = np.linalg.norm(pred - true, axis=1)
    scale = np.linalg.norm(true, axis=1) + 1e-9
    print(f"val L2 err: mean {err.mean():.3f} MW, p90 {np.percentile(err,90):.3f}, "
          f"rel {(err/scale).mean():.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
