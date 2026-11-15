"""Finalize generated numeric macros without changing any numerical value.

Run after code/analyze_results.py. Digit-containing macro names are made safe
for TeX, and the I/O retry macro is derived from the retained retry records.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manuscript" / "generated" / "numbers.tex"
RETRIES = ROOT / "results" / "io_retries.json"

lines = []
for line in PATH.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"\\newcommand\{\\(Num\w*?\d\w*)\}\{(.*)\}", line)
    if match:
        name, value = match.groups()
        line = f"\\expandafter\\newcommand\\csname {name}\\endcsname{{{value}}}"
    if "NumIORetryCount" not in line:
        lines.append(line)

retry_data = json.loads(RETRIES.read_text(encoding="utf-8"))
records = retry_data["records"]
assert all(record["status"] == "complete" for record in records)
lines.append(f"\\newcommand{{\\NumIORetryCount}}{{{len(records)}}}")
PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Attach the verification provenance and retry metadata to the analysis report.
analysis_path = ROOT / "results" / "analysis.json"
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
analysis["macros"]["IORetryCount"] = str(len(records))
analysis["io_retry_count"] = len(records)
verification = json.loads((ROOT / "results" / "verification.json").read_text(encoding="utf-8"))
verification_keys = (
    "protocol_sha256", "expected_runs", "verified_runs", "instances",
    "standalone_ml_vs_hybrid_ml", "exact_tail_replay", "source_and_raw_hashes",
    "residual_replay", "certificate_replay",
)
analysis["verification"] = {key: verification[key] for key in verification_keys}
analysis["structural_verification"] = json.loads(
    (ROOT / "results" / "structural_verification.json").read_text(encoding="utf-8")
)
analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
print(f"Finalized {PATH} and analysis provenance; I/O retry count = {len(records)}")
