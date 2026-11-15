"""Run one fingerprinted validation instance and save raw states + certificate."""
import argparse, datetime, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import data, admm
from rnn import ZProxy, RNNPredictor
from certificates import evaluate_certificate


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--algo", choices=["exact", "ml", "hybrid"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, default=50.0)
    ap.add_argument("--max-ml", type=int, default=100)
    ap.add_argument("--warm", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.algo}_S{args.S}_seed{args.seed}"
    case = data.load_case118()
    scen = data.make_scenarios(case, S=args.S, T=24, seed=args.seed)
    prob = admm.Problem(case, scen, S=args.S, T=24, rho=args.rho)
    pred = None
    if args.algo in ("ml", "hybrid"):
        net = ZProxy()
        pred = RNNPredictor(prob, net)
        pred.load_checkpoint(str(Path(__file__).with_name("proxy_clean.pt")))
    start = time.perf_counter()
    if args.algo == "exact":
        res = admm.run_exact(prob, max_iter=2000, hist=True, verbose=True,
                             warm=args.warm)
        terminal = res
        K_ml, K_tail = 0, res["K"]
        history = res["history"]
        comp = np.empty((0, 3))
        stop = "converged" if res["converged"] else "iteration_budget"
    elif args.algo == "ml":
        res = admm.run_ml(prob, pred, K_budget=args.max_ml, hist=True, verbose=True)
        terminal = res
        K_ml, K_tail = res["K_ml"], 0
        history = res["history"]
        comp = res["component_times"]
        stop = res["stop_reason"]
    else:
        res = admm.run_hybrid(prob, pred, K_budget=args.max_ml, verbose=True,
                              warm=args.warm)
        terminal = res["tail"]
        K_ml, K_tail = res["ml"]["K_ml"], res["tail"]["K"]
        history = res["tail"]["history"]
        comp = res["ml"]["component_times"]
        stop = "converged" if res["tail"]["converged"] else "iteration_budget"
    algorithm_wall = time.perf_counter() - start
    cert, carr = evaluate_certificate(prob, terminal["x"], terminal["lam"])
    if cert["dual_scenario_min_margin"] < -1e-6:
        raise RuntimeError("Rejected certificate: scenario dual exceeds primal")
    if cert["V_primal_dual_spread"] < -1e-5:
        raise RuntimeError("Rejected certificate: pinned dual exceeds primal")
    if args.algo in ("exact", "hybrid") and not terminal["converged"]:
        raise RuntimeError("Exact/hybrid run did not satisfy both stopping tolerances")
    qp_infos = terminal.get("qp_infos", [])
    summary = {
        "S": args.S, "seed": args.seed, "rho": args.rho, "T": 24,
        "algorithm": args.algo, "K_ml": K_ml, "K_tail": K_tail,
        "iterations_total": K_ml + K_tail,
        "r_terminal": terminal.get("r", terminal.get("r_plateau")),
        "s_terminal": terminal.get("s", terminal.get("s_plateau")),
        "algorithm_wall_measured": algorithm_wall,
        "algorithm_wall_function": res["wall"],
        "stop_reason": stop,
        "ml_component_mean_seconds": comp.mean(axis=0).tolist() if len(comp) else None,
        "ml_component_sum_seconds": comp.sum(axis=0).tolist() if len(comp) else None,
        "ml_other_seconds": float(res["ml"]["t_other"]) if args.algo == "hybrid" else (float(res["t_other"]) if args.algo == "ml" else None),
        "inner_qp_worst": {
            "res": max((q["res"] for q in qp_infos), default=None),
            "mu": max((q["mu"] for q in qp_infos), default=None),
            "iters": max((q["iters"] for q in qp_infos), default=None)},
        "certificate": cert,
    }
    arrays = {
        "x": terminal["x"], "x_prev": terminal["x_prev"], "z0": terminal["z0"],
        "lam": terminal["lam"], "history": history,
        "component_times": comp,
        "dual_p": carr["dual_p"], "dual_sh": carr["dual_sh"],
        "dual_cur": carr["dual_cur"], "pinned_tail_p": carr["pinned_tail_p"],
        "dual_values": carr["dual_values"],
        "dual_primal_values": carr["dual_primal_values"],
    }
    if "p" in terminal: arrays["terminal_full_p"] = terminal["p"]
    if args.algo == "hybrid":
        arrays["ml_history"] = res["ml"]["history"]
        arrays["ml_x"] = res["ml"]["x"]
        arrays["ml_z0"] = res["ml"]["z0"]
        arrays["ml_lam"] = res["ml"]["lam"]
    np.savez_compressed(out / f"{stem}.npz", **arrays)
    (out / f"{stem}.json").write_text(json.dumps(clean(summary), indent=2))
    code_dir = Path(__file__).parent
    fingerprints = {p.name: sha(p) for p in sorted(code_dir.iterdir())
                    if p.is_file() and p.suffix in (".py", ".m", ".pt")}
    manifest = {
        "run": stem, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "config": vars(args),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "threads": {k: os.environ.get(k) for k in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "source_sha256": fingerprints,
        "raw_npz_sha256": sha(out / f"{stem}.npz"),
        "summary_sha256": sha(out / f"{stem}.json"),
        "certificate_scope": "Floating-point scenario dual lower bound and independently solved pinned-recourse upper value; solver diagnostics included.",
    }
    (out / f"{stem}_manifest.json").write_text(json.dumps(clean(manifest), indent=2))
    print("SUMMARY", json.dumps(clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
