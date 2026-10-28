"""Topic files: one official state-exam topic and its reference text per Markdown file.

A file starts with a YAML header between two '---' lines; the reference text follows:

    ---
    topic_id: SYSTEM-2022-1
    domain: SYSTEM
    title: Systems fundamentals and control loop basics
    subquestions: [...]
    provenance: {...}
    ---
    The reference text, in Markdown.

The text's SHA-256 is computed here, never read from the header, so a stored hash always describes what the
endpoints received. The reference texts are not in this repository (copyright); a study names their folder.
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REQUIRED = ("topic_id", "domain", "title")
DOMAIN_ORDER = {"NETCOMP": 0, "SPECALK": 1, "SYSTEM": 2}


class TopicFileError(ValueError):
    pass


@dataclass(frozen=True)
class Topic:
    topic_id: str
    domain: str
    title: str
    text: str
    meta: dict = field(default_factory=dict, compare=False)   # the rest of the header

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def read_topic(path: Path | str) -> Topic:
    path = Path(path)
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.match(r"---\n(.*?)\n---\n(.*)\Z", raw, re.S)
    if not match:
        raise TopicFileError(f"{path.name}: no YAML header between '---' lines")
    header = yaml.safe_load(match.group(1)) or {}
    missing = [key for key in REQUIRED if not header.get(key)]
    if missing:
        raise TopicFileError(f"{path.name}: header misses {missing}")
    if path.stem != header["topic_id"]:
        raise TopicFileError(f"{path.name}: the file name must be the topic_id ({header['topic_id']}.md)")
    text = match.group(2).strip()
    if not text:
        raise TopicFileError(f"{path.name}: the reference text is empty")
    meta = {k: v for k, v in header.items() if k not in REQUIRED}
    return Topic(str(header["topic_id"]), str(header["domain"]), str(header["title"]), text, meta)


def load_topics(folder: Path | str, ids: list[str] | None = None) -> list[Topic]:
    """The topics in `folder` (all of them, or `ids`), in a stable order: domain, then topic number."""
    folder = Path(folder)
    if not folder.is_dir():
        raise TopicFileError(f"topic folder not found: {folder}")
    topics = {t.topic_id: t for t in map(read_topic, sorted(folder.glob("*.md")))}
    if ids is not None:
        unknown = sorted(set(ids) - set(topics))
        if unknown:
            raise TopicFileError(f"no topic file for {unknown} in {folder}")
        topics = {i: topics[i] for i in ids}
    return sorted(topics.values(), key=_order)


def _order(topic: Topic):
    number = topic.topic_id.rsplit("-", 1)[-1]
    return DOMAIN_ORDER.get(topic.domain, 99), topic.domain, int(number) if number.isdigit() else 0, topic.topic_id
