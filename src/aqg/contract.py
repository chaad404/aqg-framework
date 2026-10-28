"""The generation contract: one prompt, identical for every endpoint, versioned and hashed.

Template mcq-v2 is the prototype's prompt v2 as its single-call conditions sent it, so the rendered prompt is
the one the prototype piloted. It asks for plain-text JSON; validate.py measures how well each endpoint keeps
to it.
"""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
DEFAULT_TEMPLATE = "mcq-v2"
ITEMS_PER_CALL = 5
LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Template:
    version: str
    system: str
    user: str           # with {CONTEXT} and {NUM_QUESTIONS} slots

    @property
    def sha256(self) -> str:
        return _sha256({"version": self.version, "system": self.system, "user": self.user})


@lru_cache
def load_template(version: str = DEFAULT_TEMPLATE) -> Template:
    def read(name: str) -> str:
        return (PROMPTS / name).read_text(encoding="utf-8").replace("\r\n", "\n")   # same hash on every OS
    return Template(version, read(f"{version}.system.txt"), read(f"{version}.txt"))


def build_messages(template: Template, context: str, num_questions: int = ITEMS_PER_CALL) -> list[dict]:
    # str.replace, not str.format: the JSON example in the template is full of braces. The count goes in
    # first, so a context can never fill the count slot.
    user = template.user.replace("{NUM_QUESTIONS}", str(num_questions)).replace("{CONTEXT}", context)
    return [{"role": "system", "content": template.system}, {"role": "user", "content": user}]


def prompt_sha256(messages: list[dict]) -> str:
    """Hash of the exact messages sent: equal hashes mean the endpoints received the same prompt."""
    return _sha256(messages)


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
