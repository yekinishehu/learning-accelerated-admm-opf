"""Restore the canonical verification layout without deleting flattened files.

The repository currently contains both flattened root-level artifacts and the
original ZIP archives. The per-run manifests fingerprint the executed sources,
so this utility never rewrites those sources. It reconstructs each path listed
in SHA256SUMS from a root-level copy or an original ZIP member, then verifies
every listed SHA-256 digest.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVES = (
    "code.zip",
    "experiments.zip",
    "training_data.zip",
    "manuscript.zip",
    "licenses.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entries() -> list[tuple[str, str]]:
    result = []
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        result.append((digest, relative.lstrip("*")))
    return result


def extract_member(relative: str, destination: Path) -> bool:
    for archive_name in ARCHIVES:
        archive_path = ROOT / archive_name
        if not archive_path.is_file():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            if relative not in archive.namelist():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(relative))
            return True
    return False


def main() -> int:
    manifest = entries()
    restored = 0
    for _, relative in manifest:
        destination = ROOT / relative
        if destination.exists():
            continue
        flattened = ROOT / destination.name
        if flattened.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flattened, destination)
            restored += 1
        elif extract_member(relative, destination):
            restored += 1

    missing = [relative for _, relative in manifest if not (ROOT / relative).is_file()]
    mismatched = [
        relative
        for expected, relative in manifest
        if (ROOT / relative).is_file() and sha256(ROOT / relative) != expected
    ]
    report = {
        "manifest_entries": len(manifest),
        "restored_paths": restored,
        "missing_paths": missing,
        "sha256_mismatches": mismatched,
    }
    print(report)
    return 1 if missing or mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
