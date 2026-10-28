# aqg-framework

A framework for generating technical multiple-choice questions (MCQs) with large language models and comparing
model endpoints on the same task. It is the software part of my BSc thesis, *A Comparative Framework for Automated
Technical Multiple-Choice Question Generation Using Large Language Models* (University of Pécs, Faculty of
Engineering and Information Technology, 2026/27).

**Status: in development.** Modules are added one at a time, each with its tests, and this README follows them.
Done so far: secret handling with a pre-commit guard, and the endpoint registry and client (see Endpoints).

## What it does (planned)

- reads structured syllabus topics: the official topic description plus a verified reference text;
- sends one fixed generation contract to several LLM endpoints and stores every request and response with its
  provenance (model ID, host, settings, tokens, latency, cost);
- checks every question with deterministic structural, schema and item-writing rules;
- scores questions with reproducible metrics: semantic alignment to the source, distractor plausibility and
  diversity, readability, latency, tokens and cost;
- runs repeated, resumable experiments and analyses them per topic.

## Layout

| Path | Purpose |
|---|---|
| `src/aqg/` | the package |
| `configs/` | the pinned endpoints (`endpoints.yaml`) |
| `tests/` | automated tests; they need no network and no API keys |
| `tools/` | developer tools, such as the pre-commit secret guard |

## Endpoints

`configs/endpoints.yaml` pins each compared endpoint: the model ID, the host that serves it, the settings
sent with every request (the same for all endpoints) and the prices used for cost estimates. An alias is
never re-pointed; a different model or host gets a new alias.

Before every run, check that all endpoints are still offered (this only lists models and costs nothing):

```
python -m aqg.endpoints.check
```

Every call through `EndpointClient` returns a `Completion` that records the request, the raw response,
the returned model ID, the host that answered, tokens, billed and estimated cost, latency and every retry.
Failed calls raise `EndpointError` with all their attempts, so failures can be counted rather than hidden.

## Setup

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate            # Windows (on Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
git config core.hooksPath .githooks
python -m pytest
```

API keys are read from environment variables, or from a `.env` file that git ignores (see `src/aqg/config.py`).
The pre-commit hook (`tools/check_staged.py`) refuses any commit that contains a key, a database, a PDF or local
data.

## Data

Course materials and the topic reference texts are not part of this repository (copyright). The repository holds
the code and the experiment protocol. The generated questions and their scores will be published after the thesis
is assessed.

## License

MIT, see `LICENSE`.
