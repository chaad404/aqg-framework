import json

import pytest

from aqg.validate import (balance_keys, check_item, check_reply, evidence_score, item_flags, parse_reply,
                          repair_latex_escapes)

STEM = "Which property of a closed-loop control system allows it to correct the effect of disturbances on the output?"


def item(**changes):
    base = {"question": STEM,
            "options": {"A": "Feedback of the measured output", "B": "A larger actuator gain",
                        "C": "Feedforward of the setpoint value", "D": "An open-loop reference model"},
            "answer": "A", "evidence": "Control action depends on the system output (feedback)."}
    base.update(changes)
    return base


def reply(n=5, **changes):
    return json.dumps({"questions": [item(**changes) for _ in range(n)]})


# --- parsing --------------------------------------------------------------------------------------------

def test_strict_reply():
    parsed = parse_reply(reply())
    assert (parsed.mode, parsed.strict_ok, parsed.repairs) == ("strict", True, [])


def test_code_fence_and_prose_are_repaired_and_recorded():
    fenced = parse_reply("```json\n" + reply() + "\n```")
    assert (fenced.mode, fenced.strict_ok, fenced.repairs) == ("fenced", False, ["fenced"])
    prose = parse_reply("Here are the questions:\n" + reply() + "\nGood luck with the exam! {not json}")
    assert (prose.mode, prose.repairs) == ("extracted", ["extracted"]) and len(prose.data["questions"]) == 5


def test_truncated_reply_fails_with_the_json_error():
    parsed = parse_reply(reply()[:-40])
    assert parsed.data is None and parsed.mode == "failed" and "JSONDecodeError" in parsed.error


@pytest.mark.parametrize("written,meant", [
    (r"\frac{1}{s}", r"\frac{1}{s}"),            # valid JSON, but a form feed + "rac" if not repaired
    (r"\theta and \tau", r"\theta and \tau"),    # tab + "heta"
    (r"\omega_n", r"\omega_n"),                  # invalid JSON escape
    (r"\( x(t) \)", r"\( x(t) \)"),
    (r"\\frac{1}{s}", r"\frac{1}{s}"),           # already escaped: unchanged
    ("line\\nThe next", "line\nThe next"),        # a real newline stays a newline
    (r"don\'t", "don't"),
])
def test_latex_escapes(written, meant):
    text = '{"q": "' + written + '"}'
    assert json.loads(repair_latex_escapes(text))["q"] == meant
    assert repair_latex_escapes(repair_latex_escapes(text)) == repair_latex_escapes(text)   # idempotent


def test_latex_repair_is_recorded_even_when_the_reply_was_valid_json():
    parsed = parse_reply('{"questions": [], "note": "H(s) = \\frac{1}{s}"}')
    assert parsed.strict_ok and parsed.repairs == ["latex_escapes"] and parsed.data["note"] == r"H(s) = \frac{1}{s}"


# --- schema ---------------------------------------------------------------------------------------------

def test_clean_reply_is_accepted_and_strict():
    check = check_reply(reply())
    assert check.accepted and check.strict_ok and check.errors == []


@pytest.mark.parametrize("answer,repair", [("b", "answer_format"), ("B)", "answer_format"),
                                           ("B. A larger actuator gain", "answer_format"),
                                           ("Option B", "answer_format"),
                                           ("A larger actuator gain", "answer_as_text")])
def test_unambiguous_answer_forms_are_repaired(answer, repair):
    result = check_item(item(answer=answer))
    assert result.item["answer"] == "B" and result.repairs == [repair] and not result.strict_ok


def test_answer_text_starting_with_a_is_not_read_as_label_a():
    result = check_item(item(answer="A different statement that is no option"))
    assert result.item is None and "not one of A-D" in result.errors[0]


def test_options_as_list_and_own_label_prefixes_are_repaired():
    result = check_item(item(options=["A) Feedback of the measured output", "B) A larger actuator gain only",
                                      "C) Feedforward of the setpoint value", "D) An open-loop reference model"]))
    assert result.repairs == ["options_list", "option_label_prefix"]
    assert result.item["options"]["C"] == "Feedforward of the setpoint value"


def test_numeric_options_become_text():
    result = check_item(item(options={"A": 2, "B": 4, "C": 8.5, "D": "16"}, answer="C"))
    assert result.item["options"] == {"A": "2", "B": "4", "C": "8.5", "D": "16"}
    assert result.repairs == ["options_numbers"] and not result.strict_ok


@pytest.mark.parametrize("changes,error", [
    ({"options": {"A": "x1", "B": "x2", "C": "x3"}}, "exactly A, B, C and D"),
    ({"options": {"A": "Same", "B": "same", "C": "x3", "D": "x4"}}, "identical"),
    ({"options": {"A": "x1", "B": " ", "C": "x3", "D": "x4"}}, "option B is empty"),
    ({"answer": "E"}, "not one of A-D"),
    ({"answer": None}, "no answer"),
    ({"question": ""}, "no question"),
])
def test_unusable_items(changes, error):
    result = check_item(item(**changes))
    assert result.item is None and any(error in e for e in result.errors)


def test_wrong_count_rejects_the_reply():
    check = check_reply(reply(n=4))
    assert not check.accepted and check.errors == ["expected 5 questions, got 4"]


def test_one_unusable_item_rejects_the_reply():
    data = json.loads(reply())
    data["questions"][2]["answer"] = "E"
    check = check_reply(json.dumps(data))
    assert not check.accepted and check.errors[0].startswith("item 3:")


def test_list_without_wrapper_is_repaired():
    check = check_reply(json.dumps([item()] * 5))
    assert check.accepted and not check.strict_ok and check.batch_repairs == ["top_level_list"]


def test_missing_evidence_keeps_the_item_but_is_not_strict():
    result = check_item(item(evidence=None))
    assert result.item["evidence"] == "" and not result.strict_ok and result.errors == []


# --- item-writing flags ---------------------------------------------------------------------------------

def test_clean_item_has_no_flags():
    assert item_flags(check_item(item()).item) == []


@pytest.mark.parametrize("changes,flag", [
    ({"options": {"A": "Feedback of the measured output", "B": "All of the above",
                  "C": "Feedforward of the setpoint value", "D": "An open-loop reference model"}},
     "all_or_none_of_the_above"),
    ({"question": "Which of the following is NOT a property of a closed-loop control system that corrects disturbances?"},
     "negative_stem"),
    ({"options": {"A": "Feedback of the measured output", "B": "It always amplifies the actuator gain",
                  "C": "Feedforward of the setpoint value", "D": "An open-loop reference model"}},
     "absolute_in_distractors:B"),
    ({"options": {"A": "Feedback of the measured output to the controller, compared with the setpoint",
                  "B": "A larger actuator gain", "C": "A setpoint feedforward", "D": "A reference model"}},
     "key_longest"),
    ({"question": "What corrects disturbances?"}, "stem_length"),
    ({"question": "According to the text, which property of a closed-loop control system allows it to correct disturbances?"},
     "mentions_source"),
    ({"options": {"A": "Feedback of the measured output", "B": "The feedback of the measured-output.",
                  "C": "Feedforward of the setpoint value", "D": "An open-loop reference model"}},
     "near_identical:A,B"),
])
def test_item_writing_flags(changes, flag):
    assert flag in item_flags(check_item(item(**changes)).item)


@pytest.mark.parametrize("options", [
    {"A": "ζ = 1", "B": "ζ < 1", "C": "ζ = 2", "D": "ζ > 1"},                    # seen in the pilot
    {"A": "+90°", "B": "-90°", "C": "0°", "D": "+45°"},
    {"A": "The sender encrypts with the receiver's key", "B": "The receiver encrypts with the sender's key",
     "C": "Both use one shared key", "D": "No key is used at all"},              # a swapped order is a real distractor
])
def test_symbols_and_word_order_make_options_different(options):
    assert not any(f.startswith("near_identical") for f in item_flags(check_item(item(options=options)).item))


def test_context_switch_and_in_the_context_of_are_not_source_mentions():
    stem = ("In the context of process scheduling, which event makes the operating system perform the context "
            "switch between two user processes?")
    assert "mentions_source" not in item_flags(check_item(item(question=stem)).item)
    stem = "Which part of the address space of a process holds the machine code, called the text segment, at run time?"
    assert "mentions_source" not in item_flags(check_item(item(question=stem)).item)


def test_cannot_makes_a_negative_stem():
    stem = "Which property of a closed-loop control system cannot be achieved by an open-loop controller alone?"
    assert "negative_stem" in item_flags(check_item(item(question=stem)).item)


# --- evidence -------------------------------------------------------------------------------------------

REFERENCE = ("### 2.1 Open loop\n- **Definition**: Control action is independent of the system output.\n"
             "- **Closed loop**: Control action depends on the system output (feedback).\n"
             "- **Transfer function**: \\( H(s) = \\frac{Y(s)}{X(s)} \\) (Laplace domain).")


def test_verbatim_and_reformatted_evidence_match():
    assert evidence_score("Control action depends on the system output (feedback).", REFERENCE) == 100
    assert evidence_score("Closed loop: control action depends on the system output", REFERENCE) >= 90
    assert evidence_score("Transfer function: H(s) = \\frac{Y(s)}{X(s)}", REFERENCE) >= 90


def test_invented_evidence_and_weakest_fragment():
    assert evidence_score("A PID controller removes every steady-state error in all plants.", REFERENCE) < 90
    joined = "Control action is independent of the system output ... the integrator removes the offset entirely"
    assert evidence_score(joined, REFERENCE) < 90
    assert evidence_score("", REFERENCE) == 0 and evidence_score("too short", REFERENCE) == 0


# --- key balancing --------------------------------------------------------------------------------------

def test_balancing_is_seeded_and_balanced_and_keeps_the_key():
    items = [check_item(item()).item for _ in range(5)]
    first = balance_keys(items, "SYSTEM-2022-1", "gpt-oss", 1)
    assert first == balance_keys(items, "SYSTEM-2022-1", "gpt-oss", 1)
    assert first != balance_keys(items, "SYSTEM-2022-1", "gpt-oss", 2)
    assert set(q["answer"] for q in first) == {"A", "B", "C", "D"}           # 5 items: every letter at least once
    for original, moved in zip(items, first):
        assert moved["options"][moved["answer"]] == original["options"][original["answer"]]
        assert moved["raw_answer"] == "A"
        assert {label: original["options"][src] for label, src in zip("ABCD", moved["permutation"])} == moved["options"]


def test_balancing_over_many_calls_is_close_to_uniform():
    items = [check_item(item()).item for _ in range(5)]
    counts = {label: 0 for label in "ABCD"}
    for call in range(400):
        for q in balance_keys(items, "topic", "endpoint", call):
            counts[q["answer"]] += 1
    assert all(400 <= n <= 600 for n in counts.values())     # 2,000 items, 500 expected per letter
