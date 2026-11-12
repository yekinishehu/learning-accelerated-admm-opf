"""Execute the frozen strengthened-cold-v1 grid without changing solver sources."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RUNS = ROOT / "results" / "runs"
PROTOCOL = json.loads((ROOT / "strengthening_protocol.json").read_text())


def complete(S, seed, algo):
    stem = f"{algo}_S{S}_seed{seed}"
    return all((RUNS / f"{stem}{suffix}").exists()
               for suffix in (".npz", ".json", "_manifest.json"))


def main():
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    failures = []
    for S in PROTOCOL["scenario_counts"]:
        for seed in PROTOCOL["seeds"]:
            for algo in PROTOCOL["algorithms"]:
                stem = f"{algo}_S{S}_seed{seed}"
                if complete(S, seed, algo):
                    print(f"SKIP complete {stem}", flush=True)
                    continue
                command = [sys.executable, "run_one.py", "--S", str(S),
                           "--seed", str(seed), "--algo", algo,
                           "--out", str(RUNS), "--rho", "0.01"]
                print("START", stem, datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      flush=True)
                proc = subprocess.run(command, cwd=CODE, env=env)
                if proc.returncode != 0:
                    failure = {
                        "run": stem,
                        "returncode": proc.returncode,
                        "command": command,
                        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "policy": "Retained as a failed run; no seed replacement or parameter tuning."
                    }
                    failures.append(failure)
                    (RUNS / f"{stem}_FAILURE.json").write_text(json.dumps(failure, indent=2))
                    print("FAILED", stem, proc.returncode, flush=True)
                else:
                    print("DONE", stem, flush=True)
    status = {"protocol": PROTOCOL["protocol_version"], "failures": failures,
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (ROOT / "results" / "strengthened_execution.json").write_text(json.dumps(status, indent=2))
    print("EXECUTION_FINISHED", json.dumps(status), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
