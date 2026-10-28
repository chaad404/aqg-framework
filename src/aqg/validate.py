"""Checks one endpoint reply against the generation contract: parse -> schema -> flags -> evidence -> key balancing.

Every deviation is recorded. The strict results feed the structural-consistency measures (RQ1). Deviations
whose meaning is unambiguous are repaired the same way for every endpoint, so a usable reply is not thrown
away; a reply that cannot be repaired is rejected and the call is repeated (run.py).

Item-writing flags and the evidence score never remove an item: they are measurements, and the human rating
study tests how far they go.
"""
import hashlib
import json
import random
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from aqg.contract import ITEMS_PER_CALL, LABELS

# --- parsing -------------------------------------------------------------------------------------------

FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)```", re.S)

# LaTeX commands whose first letter makes a valid JSON escape (\b \f \n \r \t). "\frac" is valid JSON (a form
# feed, then "rac"), so LaTeX written with single backslashes is silently corrupted. A backslash followed by
# one of these names and then a non-letter is read as LaTeX.
LATEX_COMMANDS = (
    "backslash", "bar", "begin", "beta", "bf", "big", "bigcap", "bigcup", "bigg", "bigl", "bigr", "binom",
    "bmod", "boldsymbol", "bot", "boxed", "breve", "bullet",
    "flat", "forall", "frac", "frown",
    "nabla", "ne", "neg", "neq", "newline", "ngeq", "ni", "nleq", "nmid", "nonumber", "not", "notin", "nu",
    "rangle", "rbrace", "rceil", "rfloor", "rho", "right", "rightarrow", "rm",
    "tag", "tan", "tanh", "tau", "text", "textbf", "textit", "textrm", "textstyle", "texttt", "tfrac",
    "therefore", "theta", "tilde", "times", "to", "top", "triangle", "tt",
)
_BACKSLASH = re.compile(r"""\\(["\\/]|u[0-9a-fA-F]{4}|'|(?:%s)(?![A-Za-z])|[bfnrt]|)"""
                        % "|".join(sorted(LATEX_COMMANDS, key=len, reverse=True)))


def repair_latex_escapes(text: str) -> str:
    """Escapes the backslashes of LaTeX commands and of invalid JSON escapes; valid escapes stay as they are."""
    def fix(match):
        tail = match.group(1)
        if tail == "'":
            return "'"                              # \' is not JSON; the quote needs no escape
        if tail[:1] in ('"', "\\", "/") or (tail[:1] == "u" and len(tail) == 5) or tail in ("b", "f", "n", "r", "t"):
            return match.group(0)                   # a valid escape
        return "\\\\" + tail                         # a LaTeX command or an invalid escape such as \omega or \(
    return _BACKSLASH.sub(fix, text)


@dataclass
class Parsed:
    data: object | None          # the parsed JSON value, or None
    mode: str                    # where the JSON was found: strict (the whole reply), fenced, extracted, failed
    strict_ok: bool              # the reply as sent is JSON, without any repair
    repairs: list[str] = field(default_factory=list)   # e.g. ["fenced", "latex_escapes"]
    error: str | None = None


def parse_reply(text: str) -> Parsed:
    raw = (text or "").strip()
    strict_value, strict_error = _decode(raw, whole=True)
    for mode, candidate in _candidates(raw):
        repaired = repair_latex_escapes(candidate)
        data, _ = _decode(repaired, whole=mode != "extracted")
        if data is not None:
            repairs = ([] if mode == "strict" else [mode]) + (["latex_escapes"] if repaired != candidate else [])
            return Parsed(data, mode, strict_value is not None, repairs)
    return Parsed(None, "failed", False, error=strict_error)


def _candidates(raw: str):
    yield "strict", raw
    fenced = FENCE.search(raw)
    if fenced:
        yield "fenced", fenced.group(1).strip()
    # JSON inside prose: decode the first complete value from the first bracket on, ignoring the text after it
    for start in sorted(i for i in (raw.find("{"), raw.find("[")) if i >= 0):
        yield "extracted", raw[start:]


def _decode(text: str, whole: bool) -> tuple[object | None, str | None]:
    try:
        value = json.loads(text) if whole else json.JSONDecoder().raw_decode(text)[0]
    except (ValueError, RecursionError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (value, None) if isinstance(value, (dict, list)) else (None, "the reply is JSON but not an object")


# --- schema --------------------------------------------------------------------------------------------

STEM_KEYS = ("question", "question_stem", "stem")
ANSWER_KEYS = ("answer", "correct_answer", "correct_option", "correct")
_OWN_LABEL = re.compile(r"^\(?([A-D])[).:]\s+")


@dataclass
class ItemCheck:
    item: dict | None                # {"question", "options": {A..D}, "answer", "evidence"}; None if unusable
    strict_ok: bool                  # exactly the requested shape: nothing repaired, nothing missing
    repairs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def check_item(raw) -> ItemCheck:
    if not isinstance(raw, dict):
        return ItemCheck(None, False, errors=[f"the item is a {type(raw).__name__}, not an object"])
    repairs, errors = [], []

    stem_key = next((k for k in STEM_KEYS if k in raw), None)
    stem = raw.get(stem_key) if stem_key else None
    if stem_key not in (None, "question"):
        repairs.append("question_key")
    if not isinstance(stem, str) or not stem.strip():
        errors.append("no question")

    options = raw.get("options")
    if isinstance(options, list) and len(options) == len(LABELS):
        options = dict(zip(LABELS, options))
        repairs.append("options_list")
    if isinstance(options, dict) and sorted(map(str, options)) == ["a", "b", "c", "d"]:
        options = {str(k).upper(): v for k, v in options.items()}
        repairs.append("options_lowercase_labels")
    if not isinstance(options, dict) or sorted(options) != list(LABELS):
        errors.append("options must be exactly A, B, C and D")
        options = None
    else:
        options = {label: options[label] for label in LABELS}
        for label, text in options.items():
            if isinstance(text, (int, float)) and not isinstance(text, bool):
                text = options[label] = str(text)          # a numeric option written as a JSON number
                if "options_numbers" not in repairs:
                    repairs.append("options_numbers")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"option {label} is empty")
                continue
            own = _OWN_LABEL.match(text.strip())
            if own and own.group(1) == label:
                options[label] = text.strip()[own.end():]
                if "option_label_prefix" not in repairs:
                    repairs.append("option_label_prefix")
            options[label] = options[label].strip()
        if not errors and len({text.lower() for text in options.values()}) != len(LABELS):
            errors.append("two options are identical")

    answer_key = next((k for k in ANSWER_KEYS if raw.get(k) not in (None, "")), None)
    if answer_key is None:
        errors.append("no answer")
        answer = None
    else:
        if answer_key != "answer":
            repairs.append("answer_key")
        answer, how = _answer_label(raw[answer_key], options)
        if answer is None:
            errors.append(f"answer {str(raw[answer_key])[:40]!r} is not one of A-D")
        elif how:
            repairs.append(how)

    evidence = raw.get("evidence")
    if isinstance(evidence, list) and evidence and all(isinstance(e, str) for e in evidence):
        evidence = " ... ".join(evidence)
        repairs.append("evidence_list")
    evidence = evidence.strip() if isinstance(evidence, str) else ""
    if errors:
        return ItemCheck(None, False, repairs, errors)
    item = {"question": stem.strip(), "options": options, "answer": answer, "evidence": evidence}
    return ItemCheck(item, not repairs and bool(evidence), repairs, errors)


def _answer_label(value, options) -> tuple[str | None, str | None]:
    """(label, repair) for an answer given as "B", or unambiguously as "b", "B)", "B. text", "Option B" or the
    option's own text."""
    text = str(value).strip()
    if text in LABELS:
        return text, None
    if options and isinstance(value, str):
        hits = [label for label, option in options.items() if isinstance(option, str)
                and option.strip().lower() == text.lower()]
        if len(hits) == 1:
            return hits[0], "answer_as_text"
    match = re.match(r"^(?:option\s+)?\(?([A-Da-d])(?:$|[).:])", text, re.I)   # never "A router ..."
    return (match.group(1).upper(), "answer_format") if match else (None, None)


@dataclass
class ReplyCheck:
    parse: Parsed
    batch_repairs: list[str] = field(default_factory=list)
    batch_errors: list[str] = field(default_factory=list)
    items: list[ItemCheck] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Usable: parsed, the requested number of items, and every item usable."""
        return (self.parse.data is not None and not self.batch_errors
                and bool(self.items) and all(i.item is not None for i in self.items))

    @property
    def strict_ok(self) -> bool:
        """The reply follows the contract exactly, with nothing repaired."""
        return (self.accepted and self.parse.strict_ok and not self.parse.repairs and not self.batch_repairs
                and all(i.strict_ok for i in self.items))

    @property
    def errors(self) -> list[str]:
        out = [] if self.parse.data is not None else [f"parse: {self.parse.error}"]
        out += self.batch_errors
        out += [f"item {n}: {e}" for n, i in enumerate(self.items, 1) for e in i.errors]
        return out


def check_reply(text: str, expected: int = ITEMS_PER_CALL) -> ReplyCheck:
    parsed = parse_reply(text)
    result = ReplyCheck(parsed)
    data = parsed.data
    if data is None:
        return result
    if isinstance(data, list):
        questions = data
        result.batch_repairs.append("top_level_list")
    elif isinstance(data.get("questions"), list):
        questions = data["questions"]
    else:
        lists = [k for k, v in data.items() if isinstance(v, list)]
        if len(lists) != 1:
            result.batch_errors.append("no 'questions' list")
            return result
        questions = data[lists[0]]
        result.batch_repairs.append(f"questions_key:{lists[0]}")
    if len(questions) != expected:
        result.batch_errors.append(f"expected {expected} questions, got {len(questions)}")
    result.items = [check_item(q) for q in questions]
    return result


# --- item-writing flags (S4) ---------------------------------------------------------------------------

STEM_WORDS = (15, 35)
LENGTH_CUE_RATIO = 1.2
ABSOLUTES = {"always", "never", "all", "none", "only", "every", "completely", "entirely", "absolutely"}
ARTICLES = {"a", "an", "the"}
ALL_NONE = re.compile(r"\b(all|none)\s+of\s+(the\s+)?(above|these|options|answers)\b", re.I)
NEGATIVE_STEM = re.compile(r"\b(not|except|cannot)\b|n't\b|\b(is|are)\s+(false|incorrect|untrue)\b", re.I)
FIGURE_MENTION = re.compile(   # textbook passages cite figures and equations the question cannot show
    r"\b(fig(?:ure)?s?\.?\s*\d|tables?\s+\d+[.\-]\d|eq(?:uation)?s?\.?\s*\(\s*\d|equation\s+\d|section\s+\d+\.\d)", re.I)
SOURCE_MENTION = re.compile(   # "the text segment" of a process or "the text field" are content, not the source
    r"\b(the|this)\s+(given\s+|provided\s+|above\s+)?(text|passage)\b"
    r"(?!\s+(segment|section|file|editor|field|mode|area|box|message|string|data))"
    r"|\b(given|provided|above)\s+context\b|\bcontext\s+(above|provided|given)\b"
    r"|\b(according\s+to|based\s+on|from|in)\s+the\s+context\b(?!\s+(of|switch))", re.I)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def item_flags(item: dict) -> list[str]:
    """Item-writing flags of one usable item. Labels in a flag ("near_identical:A,C") name stored positions."""
    stem, options, key = item["question"], item["options"], item["answer"]
    others = [label for label in LABELS if label != key]
    flags = []
    if any(ALL_NONE.search(text) for text in options.values()):
        flags.append("all_or_none_of_the_above")
    if NEGATIVE_STEM.search(stem):
        flags.append("negative_stem")
    absolute = [label for label in others if ABSOLUTES & set(_words(options[label]))]
    if absolute and not ABSOLUTES & set(_words(options[key])):
        flags.append("absolute_in_distractors:" + ",".join(absolute))
    lengths = {label: len(text) for label, text in options.items()}
    mean_other = sum(lengths[label] for label in others) / len(others)
    if lengths[key] == max(lengths.values()) and lengths[key] > LENGTH_CUE_RATIO * mean_other:
        flags.append("key_longest")
    n_words = len(stem.split())
    if not STEM_WORDS[0] <= n_words <= STEM_WORDS[1]:
        flags.append("stem_length")
    if SOURCE_MENTION.search(stem):
        flags.append("mentions_source")
    if FIGURE_MENTION.search(" ".join([stem, *options.values()])):
        flags.append("mentions_figure")
    plain = {label: _plain(text) for label, text in options.items()}
    same = [a for a in LABELS if any(a != b and plain[a] == plain[b] for b in LABELS)]
    if same:
        flags.append("near_identical:" + ",".join(same))
    return flags


def _plain(option: str) -> str:
    """An option without case, articles, punctuation or hyphenation. Word order and symbols are kept: "ζ < 1" and
    "ζ > 1", or a swapped order ("A before B"), are different options."""
    text = re.sub(r"(?<=\w)-(?=\w)", " ", option.lower())
    text = re.sub(r"[.,;:!?\"'()\[\]{}]", " ", text)
    return " ".join(w for w in text.split() if w not in ARTICLES)


# --- evidence (S7) -------------------------------------------------------------------------------------

EVIDENCE_THRESHOLD = 90.0
MIN_FRAGMENT_WORDS = 3


def evidence_score(evidence: str, reference: str) -> float:
    """0-100: how closely the quoted evidence occurs in the reference text. Text is compared as lower-case words,
    so Markdown and LaTeX punctuation do not count; quotes joined by '...' are checked one by one and the weakest
    counts."""
    ref = " ".join(_words(reference))
    fragments = [" ".join(_words(f)) for f in re.split(r"\.\.\.|…", evidence or "")]
    fragments = [f for f in fragments if len(f.split()) >= MIN_FRAGMENT_WORDS]
    if not fragments or not ref:
        return 0.0
    return round(min(fuzz.partial_ratio(f, ref) for f in fragments), 2)


# --- key balancing -------------------------------------------------------------------------------------

def balance_keys(items: list[dict], *seed_parts) -> list[dict]:
    """Moves each key to a seeded position so that every letter is the key equally often (up to one extra).

    Deterministic for the same seed parts, so a rerun reproduces the stored order. Each returned item keeps the
    model's own answer letter ("raw_answer") and the permutation: "CADB" means stored A is the model's C, and so on.
    """
    rng = random.Random(int(hashlib.sha256("|".join(map(str, seed_parts)).encode()).hexdigest()[:12], 16))
    return [_move_key(item, target, rng) for item, target in zip(items, _balanced_positions(len(items), rng))]


def _balanced_positions(n: int, rng: random.Random) -> list[str]:
    positions = []
    while len(positions) < n:
        block = list(LABELS)
        rng.shuffle(block)
        positions += block
    positions = positions[:n]
    rng.shuffle(positions)
    return positions


def _move_key(item: dict, target: str, rng: random.Random) -> dict:
    key = item["answer"]
    distractors = [label for label in LABELS if label != key]
    rng.shuffle(distractors)
    rest = iter(distractors)
    source = {label: (key if label == target else next(rest)) for label in LABELS}
    return {**item, "options": {label: item["options"][source[label]] for label in LABELS}, "answer": target,
            "raw_answer": key, "permutation": "".join(source[label] for label in LABELS)}
