"""Retry only I/O-incomplete runs under the frozen protocol.

The numerical entry point and every fingerprinted source remain unchanged.  The
wrapper only makes NPZ closing/visibility and hashing robust to transient file
system delays.  Existing failure records are retained.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RUNS = ROOT / "results" / "runs"
CACHE = ROOT / "results" / "npz_cache"
CACHE.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((ROOT / "strengthening_protocol.json").read_text())
sys.path.insert(0, str(CODE))
import run_one  # noqa: E402

_ORIGINAL_SAVEZ = run_one.np.savez_compressed


def wait_for_file(path, attempts=100, delay=0.05):
    path = Path(path)
    for _ in range(attempts):
        if path.is_file():
            return path
        time.sleep(delay)
    raise FileNotFoundError(f"File did not become visible: {path}")


def robust_savez(path, **arrays):
    path = Path(path)
    _ORIGINAL_SAVEZ(path, **arrays)
    wait_for_file(path)
    cache = CACHE / path.name
    with path.open("rb") as source, cache.open("wb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    # Replace the direct archive with an independently copied file.  This
    # avoids the intermittent disappearance observed only for larger direct
    # NPZ writes on the output mount; the bytes are not changed.
    temporary = path.with_name(path.name + ".restore.tmp")
    with cache.open("rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    wait_for_file(path)


def robust_sha(path):
    path = wait_for_file(path)
    last_error = None
    for _ in range(100):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            wait_for_file(path)
            return digest.hexdigest()
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(0.05)
    raise last_error


def complete(S, seed, algo):
    stem = f"{algo}_S{S}_seed{seed}"
    return all((RUNS / f"{stem}{suffix}").is_file()
               for suffix in (".npz", ".json", "_manifest.json"))


def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    run_one.np.savez_compressed = robust_savez
    run_one.sha = robust_sha
    records = []
    os.chdir(CODE)
    for S in PROTOCOL["scenario_counts"]:
        for seed in PROTOCOL["seeds"]:
            for algo in PROTOCOL["algorithms"]:
                if complete(S, seed, algo):
                    continue
                stem = f"{algo}_S{S}_seed{seed}"
                argv = ["run_one.py", "--S", str(S), "--seed", str(seed),
                        "--algo", algo, "--out", str(RUNS), "--rho", "0.01"]
                record = {"run": stem, "started_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(), "command": argv,
                    "reason": "Retry of an I/O-incomplete artifact set; numerical settings unchanged."}
                print("IO_RETRY_START", stem, flush=True)
                sys.argv = argv
                try:
                    run_one.main()
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = repr(exc)
                    records.append(record)
                    print("IO_RETRY_FAILED", stem, repr(exc), flush=True)
                else:
                    raw = RUNS / f"{stem}.npz"
                    if not raw.is_file():
                        temporary = raw.with_name(raw.name + ".restore.tmp")
                        shutil.copyfile(CACHE / raw.name, temporary)
                        os.replace(temporary, raw)
                    record["status"] = "complete" if complete(S, seed, algo) else "failed"
                    if record["status"] == "failed":
                        record["error"] = "Artifact set incomplete after stable-copy restoration"
                    record["completed_utc"] = datetime.datetime.now(
                        datetime.timezone.utc).isoformat()
                    records.append(record)
                    print("IO_RETRY_DONE", stem, record["status"], flush=True)
    status = {"protocol": PROTOCOL["protocol_version"],
              "protocol_sha256": robust_sha(ROOT / "strengthening_protocol.json"),
              "records": records,
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (ROOT / "results" / "io_retries.json").write_text(json.dumps(status, indent=2))
    print("IO_RETRIES_FINISHED", json.dumps(status), flush=True)
    return 1 if any(r["status"] != "complete" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
