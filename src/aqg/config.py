"""Secrets and local settings.

A secret is read from the environment first. If it is not set there, these files are tried in order:
  1. the file named by the AQG_ENV_FILE environment variable,
  2. `.env` in the repository root (ignored by git),
  3. `../.secrets/aqg.env`, a folder next to the repository, so the keys never sit inside the git working tree.
Values are never printed or logged; `describe()` reports only which secrets are present.
"""
import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET_NAMES = ("OPENROUTER_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "OPENALEX_API_KEY")


def candidate_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """The secret files to try, in order."""
    files = [Path(os.environ["AQG_ENV_FILE"])] if os.environ.get("AQG_ENV_FILE") else []
    return files + [repo_root / ".env", repo_root.parent / ".secrets" / "aqg.env"]


def get_secret(name: str, repo_root: Path = REPO_ROOT) -> str | None:
    """The value of `name`, or None if no source defines it."""
    if os.environ.get(name):
        return os.environ[name]
    for path in candidate_files(repo_root):
        if path.is_file():
            value = dotenv_values(path).get(name)
            if value:
                return value
    return None


def require_secret(name: str, repo_root: Path = REPO_ROOT) -> str:
    """The value of `name`; fails with the places to put it (never with a value)."""
    value = get_secret(name, repo_root)
    if not value:
        places = ", ".join(str(p) for p in candidate_files(repo_root))
        raise RuntimeError(f"{name} is not set. Set it in the environment or in one of: {places}")
    return value


def describe(repo_root: Path = REPO_ROOT) -> dict[str, bool]:
    """Which known secrets are available, without revealing any value."""
    return {name: get_secret(name, repo_root) is not None for name in SECRET_NAMES}
