from aqg.contract import build_messages, load_template, prompt_sha256


def test_template_renders_count_and_context_without_touching_the_json_example():
    template = load_template()
    messages = build_messages(template, "Signals are functions of time.", 5)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert user.startswith("<context>\nSignals are functions of time.\n</context>")
    assert "Write 5 questions" in user and "{NUM_QUESTIONS}" not in user and "{CONTEXT}" not in user
    assert '{"questions": [{"question": "...", "options": {"A": "..."' in user
    assert "{FOCUS}" not in user and "\r" not in user


def test_a_context_cannot_fill_the_count_slot():
    user = build_messages(load_template(), "the literal text {NUM_QUESTIONS}", 5)[1]["content"]
    assert "the literal text {NUM_QUESTIONS}" in user


def test_hashes_are_stable_and_follow_the_content():
    template = load_template()
    assert len(template.sha256) == 64 and template.sha256 == load_template().sha256
    a = prompt_sha256(build_messages(template, "text one"))
    assert a == prompt_sha256(build_messages(template, "text one"))
    assert a != prompt_sha256(build_messages(template, "text two"))
