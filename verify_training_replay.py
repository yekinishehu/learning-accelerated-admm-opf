"""Regenerate all training artifacts and compare arrays/tensors bitwise."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    old = json.loads((RESULTS / "training_verification.json").read_text())
    originals = sorted((ROOT / "training_data").glob("shard_*.npz"))
    assert len(originals) == old["instances"] == 30
    with tempfile.TemporaryDirectory(prefix="training_replay_", dir=RESULTS) as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        checkpoint = tmp / "proxy.pt"
        subprocess.run([
            sys.executable, "datagen.py", str(data_dir), str(old["instances"]),
            str(old["scenarios_per_training_instance"]), "24",
            str(old["iterations_per_instance"]), str(old["seed0"]), str(old["rho"])
        ], cwd=CODE, env=env, check=True)
        subprocess.run([sys.executable, "train.py", str(data_dir), str(checkpoint),
                        str(old["epochs"])], cwd=CODE, env=env, check=True)
        replayed = sorted(data_dir.glob("shard_*.npz"))
        assert len(replayed) == len(originals)
        pair_count = 0
        for original, replay in zip(originals, replayed):
            assert original.name == replay.name
            with np.load(original) as a, np.load(replay) as b:
                assert set(a.files) == set(b.files) == {"feats", "labels"}
                for key in a.files:
                    assert np.array_equal(a[key], b[key]), (original.name, key)
                pair_count += len(a["labels"])
        assert pair_count == old["training_pairs"]
        a = torch.load(CODE / "proxy_clean.pt", map_location="cpu", weights_only=False)
        b = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert set(a) == set(b)
        for key in ("feat_mean", "feat_std", "lab_mean", "lab_std"):
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
        assert a["state_dict"].keys() == b["state_dict"].keys()
        for key in a["state_dict"]:
            assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key
    out = dict(old)
    out.update({
        "status": "pass",
        "dataset_replay": "bitwise array pass for all 30 shards",
        "checkpoint_replay": "bitwise tensor pass for weights and normalization arrays",
        "verification_driver_sha256": sha(Path(__file__)),
        "datagen_sha256": sha(CODE / "datagen.py"),
        "train_sha256": sha(CODE / "train.py"),
        "rnn_sha256": sha(CODE / "rnn.py"),
        "executed_checkpoint": "code/proxy_clean.pt",
        "executed_checkpoint_sha256": sha(CODE / "proxy_clean.pt")
    })
    (RESULTS / "training_verification.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
