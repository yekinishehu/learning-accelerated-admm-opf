"""Independent integrity gates for all reported runs."""
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
import data, admm
from certificates import evaluate_certificate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
CODE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    expected = [(20, 99), (50, 99), (100, 99), (50, 100), (50, 101)]
    records = []
    for S, seed in expected:
        case = data.load_case118()
        scen = data.make_scenarios(case, S=S, T=24, seed=seed)
        prob = admm.Problem(case, scen, S=S, T=24, rho=0.01)
        ml_a = np.load(RUNS / f"ml_S{S}_seed{seed}.npz")
        hy_a = np.load(RUNS / f"hybrid_S{S}_seed{seed}.npz")
        assert np.array_equal(ml_a["history"], hy_a["ml_history"])
        assert np.array_equal(ml_a["x"], hy_a["ml_x"])
        assert np.array_equal(ml_a["z0"], hy_a["ml_z0"])
        assert np.array_equal(ml_a["lam"], hy_a["ml_lam"])
        # Independent downstream replay from the raw ML state only.
        tail = admm.run_exact(prob, x0=hy_a["ml_x"], lam0=hy_a["ml_lam"],
                              z0_init=hy_a["ml_z0"], eps_r=1e-4,
                              eps_s=1e-4, max_iter=2000, warm=True, hist=True)
        assert np.array_equal(tail["x"], hy_a["x"])
        assert np.array_equal(tail["z0"], hy_a["z0"])
        assert np.array_equal(tail["lam"], hy_a["lam"])
        assert np.array_equal(tail["history"], hy_a["history"])
        for algo in ("exact", "ml", "hybrid"):
            stem = f"{algo}_S{S}_seed{seed}"
            summary = json.loads((RUNS / f"{stem}.json").read_text())
            manifest = json.loads((RUNS / f"{stem}_manifest.json").read_text())
            arr = np.load(RUNS / f"{stem}.npz")
            assert manifest["raw_npz_sha256"] == sha(RUNS / f"{stem}.npz")
            assert manifest["summary_sha256"] == sha(RUNS / f"{stem}.json")
            for name, digest in manifest["source_sha256"].items():
                assert sha(CODE / name) == digest, (stem, name)
            r = float(np.linalg.norm(arr["z0"] - arr["x"][None, :], axis=1).max())
            s = float(0.01 * np.linalg.norm(arr["x"] - arr["x_prev"]))
            assert r == summary["r_terminal"]
            assert s == summary["s_terminal"]
            if algo in ("exact", "hybrid"):
                assert r <= 1e-4 and s <= 1e-4
            cert = summary["certificate"]
            assert cert["dual_scenario_min_margin"] >= -1e-6
            assert cert["V_primal_dual_spread"] >= -1e-5
            assert cert["absolute_gap"] >= -1e-6
            assert cert["p_low"] > 0
            replay, _ = evaluate_certificate(prob, arr["x"], arr["lam"])
            for key in ("p_low", "V_pinned", "absolute_gap"):
                assert np.isclose(replay[key], cert[key], rtol=1e-11, atol=1e-7), (stem, key)
            records.append({"run": stem, "status": "pass", "r": r, "s": s,
                            "certificate_replay_max_abs_difference": max(
                                abs(replay[k] - cert[k]) for k in
                                ("p_low", "V_pinned", "absolute_gap"))})
    out = {"expected_runs": 15, "verified_runs": len(records),
           "standalone_ml_vs_hybrid_ml": "bitwise pass for all five instances",
           "exact_tail_replay": "bitwise pass for all five instances",
           "source_and_raw_hashes": "pass", "residual_replay": "exact pass",
           "certificate_replay": "pass", "records": records}
    path = ROOT / "results" / "verification.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))

if __name__ == "__main__": main()
