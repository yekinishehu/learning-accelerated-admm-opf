"""Make generated numeric macro declarations safe for names containing digits."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manuscript" / "generated" / "numbers.tex"
lines = []
for line in PATH.read_text().splitlines():
    match = re.fullmatch(r"\\newcommand\{\\(Num\w*?\d\w*)\}\{(.*)\}", line)
    if match:
        name, value = match.groups()
        line = f"\\expandafter\\newcommand\\csname {name}\\endcsname{{{value}}}"
    lines.append(line)
PATH.write_text("\n".join(lines) + "\n")
