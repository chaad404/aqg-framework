import hashlib

import pytest

from aqg.topics import TopicFileError, load_topics, read_topic

HEADER = "---\ntopic_id: {tid}\ndomain: {domain}\ntitle: A title\nprovenance: {{verification: unverified}}\n---\n"


def write(folder, tid, domain="SYSTEM", text="Signals are functions of time.\n", newline="\n"):
    path = folder / f"{tid}.md"
    path.write_bytes((HEADER.format(tid=tid, domain=domain) + text).replace("\n", newline).encode("utf-8"))
    return path


def test_reads_header_and_text_and_hashes_the_text(tmp_path):
    topic = read_topic(write(tmp_path, "SYSTEM-2022-1"))
    assert (topic.topic_id, topic.domain, topic.title) == ("SYSTEM-2022-1", "SYSTEM", "A title")
    assert topic.text == "Signals are functions of time."
    assert topic.meta == {"provenance": {"verification": "unverified"}}
    assert topic.sha256 == hashlib.sha256(b"Signals are functions of time.").hexdigest()


def test_windows_line_endings_give_the_same_hash(tmp_path):
    (tmp_path / "crlf").mkdir()
    a = read_topic(write(tmp_path, "SYSTEM-2022-1", text="Line one.\nLine two.\n"))
    b = read_topic(write(tmp_path / "crlf", "SYSTEM-2022-1", text="Line one.\nLine two.\n", newline="\r\n"))
    assert a.sha256 == b.sha256


def test_topics_are_ordered_by_domain_and_number(tmp_path):
    for tid, domain in [("SYSTEM-2022-10", "SYSTEM"), ("SYSTEM-2022-2", "SYSTEM"), ("NETCOMP-2022-3", "NETCOMP")]:
        write(tmp_path, tid, domain)
    assert [t.topic_id for t in load_topics(tmp_path)] == ["NETCOMP-2022-3", "SYSTEM-2022-2", "SYSTEM-2022-10"]
    assert [t.topic_id for t in load_topics(tmp_path, ["SYSTEM-2022-2"])] == ["SYSTEM-2022-2"]


@pytest.mark.parametrize("content,error", [
    ("no header at all", "no YAML header"),
    ("---\ndomain: SYSTEM\ntitle: x\n---\ntext", "misses ['topic_id']"),
    ("---\ntopic_id: SYSTEM-2022-9\ndomain: SYSTEM\ntitle: x\n---\ntext", "file name must be the topic_id"),
    ("---\ntopic_id: SYSTEM-2022-1\ndomain: SYSTEM\ntitle: x\n---\n   \n", "reference text is empty"),
])
def test_bad_files_are_refused(tmp_path, content, error):
    path = tmp_path / "SYSTEM-2022-1.md"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TopicFileError, match=error.replace("[", r"\[").replace("]", r"\]")):
        read_topic(path)


def test_unknown_topic_ids_are_refused(tmp_path):
    write(tmp_path, "SYSTEM-2022-1")
    with pytest.raises(TopicFileError, match="SYSTEM-2022-7"):
        load_topics(tmp_path, ["SYSTEM-2022-7"])
