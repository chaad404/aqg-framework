"""Pre-commit guard: refuse commits that would publish secrets or local material.

A commit is blocked when a staged file
  - is an environment or secrets file, a database, a PDF, or lives in data/, outputs/ or source_review/;
  - contains the value of a secret known to aqg.config (environment or secret files);
  - contains something shaped like an OpenRouter key.
Only file names and secret NAMES are printed, never a value.

Enable once per clone:  git config core.hooksPath .githooks
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqg import config  # noqa: E402

BLOCKED_PATH = re.compile(r"^(data|outputs|source_review|\.secrets)/|(^|/)[^/]*\.env$"
                          r"|\.(pdf|db|db-wal|db-shm|sqlite|faiss|pkl)$", re.IGNORECASE)
KEY_SHAPES = {"OpenRouter key": re.compile(r"sk-or-v1-[0-9a-f]{20,}")}
MIN_SECRET_LENGTH = 12  # shorter values would also match ordinary text


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    return [line for line in out.stdout.splitlines() if line]


def staged_text(path: str) -> str:
    out = subprocess.run(["git", "show", f":{path}"], capture_output=True, cwd=ROOT)
    return out.stdout.decode("utf-8", errors="ignore")


def known_secrets() -> dict[str, str]:
    values = {name: config.get_secret(name) for name in config.SECRET_NAMES}
    return {name: v for name, v in values.items() if v and len(v) >= MIN_SECRET_LENGTH}


def problems(files: list[str], read=staged_text, secrets: dict[str, str] | None = None) -> list[str]:
    """One message per problem found in the given files."""
    secrets = known_secrets() if secrets is None else secrets
    found = []
    for path in files:
        if BLOCKED_PATH.search(path.replace("\\", "/")):
            found.append(f"{path}: this kind of file must stay local")
            continue
        text = read(path)
        found += [f"{path}: contains the value of {name}" for name, value in secrets.items() if value in text]
        found += [f"{path}: contains an {label}" for label, shape in KEY_SHAPES.items() if shape.search(text)]
    return found


def main() -> int:
    found = problems(staged_files())
    if found:
        print("Commit blocked by tools/check_staged.py:")
        print("\n".join(f"  - {problem}" for problem in found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
