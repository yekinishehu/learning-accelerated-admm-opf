"""Additional full-grid schema, full-ML, and component-accounting checks."""
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RUNS = ROOT / "results" / "runs"
sys.path.insert(0, str(CODE))
import admm  # noqa: E402
import data  # noqa: E402
from rnn import ZProxy, RNNPredictor  # noqa: E402

PROTOCOL = json.loads((ROOT / "strengthening_protocol.json").read_text())
INSTANCES = [(S, seed) for S in PROTOCOL["scenario_counts"] for seed in PROTOCOL["seeds"]]
REQUIRED = {
    "x", "x_prev", "z0", "lam", "history", "component_times", "dual_p",
    "dual_sh", "dual_cur", "pinned_tail_p", "dual_values", "dual_primal_values",
}


def main():
    torch.set_num_threads(1)
    schemas = []
    ml_records = []
    components = []
    for S, seed in INSTANCES:
        case = data.load_case118()
        scen = data.make_scenarios(case, S=S, T=24, seed=seed)
        prob = admm.Problem(case, scen, S=S, T=24, rho=0.01)
        for algo in PROTOCOL["algorithms"]:
            stem = f"{algo}_S{S}_seed{seed}"
            path = RUNS / f"{stem}.npz"
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
            with np.load(path, allow_pickle=False) as arr:
                assert REQUIRED <= set(arr.files)
                assert all(np.isfinite(arr[k]).all() for k in arr.files)
                if algo in ("exact", "hybrid"):
                    assert "terminal_full_p" in arr.files
                if algo == "hybrid":
                    assert {"ml_history", "ml_x", "ml_z0", "ml_lam"} <= set(arr.files)
                schemas.append({"run": stem, "arrays": len(arr.files), "status": "pass"})
                summary = json.loads((RUNS / f"{stem}.json").read_text())
                comp = arr["component_times"]
                if len(comp):
                    assert np.array_equal(comp.mean(axis=0), summary["ml_component_mean_seconds"])
                    assert np.array_equal(comp.sum(axis=0), summary["ml_component_sum_seconds"])
                    components.append({"run": stem, "rows": len(comp), "mean_sum_exact": True})
        ml_raw = np.load(RUNS / f"ml_S{S}_seed{seed}.npz", allow_pickle=False)
        net = ZProxy()
        pred = RNNPredictor(prob, net)
        pred.load_checkpoint(str(CODE / "proxy_clean.pt"))
        torch.manual_seed(0)
        np.random.seed(0)
        replay = admm.run_ml(prob, pred, K_budget=100, hist=True)
        comparisons = {
            key: bool(np.array_equal(replay[key], ml_raw[key]))
            for key in ("history", "x", "x_prev", "z0", "lam")
        }
        assert all(comparisons.values()), (S, seed, comparisons)
        ml_records.append({"S": S, "seed": seed, "bitwise": comparisons})
    retries = json.loads((ROOT / "results" / "io_retries.json").read_text())["records"]
    out = {
        "protocol": PROTOCOL["protocol_version"],
        "schema_and_finiteness": "pass for all 45 runs",
        "schema_records": schemas,
        "full_ml_replay": "bitwise pass for all 15 instances",
        "ml_records": ml_records,
        "component_accounting": "exact mean/sum pass for all 30 ML/hybrid runs",
        "component_records": components,
        "io_retry_count": len(retries),
        "io_retry_sets_complete": all(r["status"] == "complete" for r in retries),
    }
    path = ROOT / "results" / "structural_verification.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("records")}, indent=2))


if __name__ == "__main__":
    main()
