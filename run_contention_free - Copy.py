"""Run the frozen grid in isolated, I/O-robust processes without CPU contention."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RUNS = ROOT / "results" / "runs"
PROTOCOL_PATH = ROOT / "strengthening_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
SENTINEL = ROOT / "results" / "contention_free_protocol.sha256"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_file(path, attempts=100, delay=0.05):
    path = Path(path)
    for _ in range(attempts):
        if path.is_file():
            return path
        time.sleep(delay)
    raise FileNotFoundError(f"File did not become visible: {path}")


def child_main(argv):
    sys.path.insert(0, str(CODE))
    import run_one
    original_savez = run_one.np.savez_compressed

    def robust_savez(path, **arrays):
        path = Path(path)
        original_savez(path, **arrays)
        wait_for_file(path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        wait_for_file(path)

    def robust_sha(path):
        path = wait_for_file(path)
        last_error = None
        for _ in range(100):
            try:
                value = sha(path)
                wait_for_file(path)
                return value
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.05)
        raise last_error

    run_one.np.savez_compressed = robust_savez
    run_one.sha = robust_sha
    sys.argv = ["run_one.py"] + argv
    run_one.main()


def complete(S, seed, algo):
    stem = f"{algo}_S{S}_seed{seed}"
    paths = {suffix: RUNS / f"{stem}{suffix}"
             for suffix in (".npz", ".json", "_manifest.json")}
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        manifest = json.loads(paths["_manifest.json"].read_text())
        config = manifest["config"]
        assert config["S"] == S and config["seed"] == seed
        assert config["algo"] == algo and config["rho"] == 0.01
        assert not config["warm"] and config["max_ml"] == 100
        assert manifest["raw_npz_sha256"] == sha(paths[".npz"])
        assert manifest["summary_sha256"] == sha(paths[".json"])
        for name, digest in manifest["source_sha256"].items():
            assert sha(CODE / name) == digest
    except (AssertionError, KeyError, FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def parent_main():
    RUNS.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha(PROTOCOL_PATH)
    if SENTINEL.exists():
        if SENTINEL.read_text().strip() != protocol_hash:
            raise RuntimeError("Frozen protocol changed; clean-run cache is invalid")
    elif any(RUNS.iterdir()):
        raise RuntimeError("Nonempty clean-run directory lacks a protocol sentinel")
    else:
        SENTINEL.write_text(protocol_hash + "\n")
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    records = []
    for S in PROTOCOL["scenario_counts"]:
        for seed in PROTOCOL["seeds"]:
            for algo in PROTOCOL["algorithms"]:
                stem = f"{algo}_S{S}_seed{seed}"
                if complete(S, seed, algo):
                    print("SKIP_HASH_VERIFIED", stem, flush=True)
                    continue
                args = ["--S", str(S), "--seed", str(seed), "--algo", algo,
                        "--out", str(RUNS), "--rho", "0.01"]
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--child", "--"] + args
                record = {"run": stem, "command": command,
                          "started_utc": datetime.datetime.now(
                              datetime.timezone.utc).isoformat()}
                print("CLEAN_START", stem, record["started_utc"], flush=True)
                proc = subprocess.run(command, cwd=CODE, env=env)
                record["returncode"] = proc.returncode
                record["complete"] = complete(S, seed, algo)
                record["finished_utc"] = datetime.datetime.now(
                    datetime.timezone.utc).isoformat()
                records.append(record)
                if proc.returncode or not record["complete"]:
                    failure = dict(record)
                    failure["policy"] = "Retained; no seed replacement or parameter tuning."
                    (RUNS / f"{stem}_FAILURE.json").write_text(json.dumps(failure, indent=2))
                    print("CLEAN_FAILED", stem, proc.returncode, flush=True)
                else:
                    print("CLEAN_DONE", stem, flush=True)
    status = {"protocol": PROTOCOL["protocol_version"],
              "protocol_sha256": protocol_hash,
              "execution": "isolated processes; no parallel compilation, training, or visual inspection",
              "expected_runs": PROTOCOL["expected_run_count"],
              "complete_runs": sum(complete(S, seed, algo)
                                   for S in PROTOCOL["scenario_counts"]
                                   for seed in PROTOCOL["seeds"]
                                   for algo in PROTOCOL["algorithms"]),
              "records": records,
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (ROOT / "results" / "contention_free_execution.json").write_text(json.dumps(status, indent=2))
    print("CLEAN_EXECUTION_FINISHED", json.dumps({k: v for k, v in status.items()
                                                   if k != "records"}), flush=True)
    return 0 if status["complete_runs"] == status["expected_runs"] else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        args = sys.argv[2:]
        if args and args[0] == "--":
            args = args[1:]
        child_main(args)
    else:
        raise SystemExit(parent_main())
