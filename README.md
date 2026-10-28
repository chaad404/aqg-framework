# aqg-framework

A framework for generating technical multiple-choice questions (MCQs) with large language models and comparing
model endpoints on the same task. It is the software part of my BSc thesis, *A Comparative Framework for Automated
Technical Multiple-Choice Question Generation Using Large Language Models* (University of Pécs, Faculty of
Engineering and Information Technology, 2026/27).

**Status: in development.** Modules are added one at a time, each with its tests, and this README follows them.
Done so far: secret handling with a pre-commit guard, the endpoint registry and client, the generation contract,
reply validation, topic files, the study store and the run engine. Next: the quality metrics and the analysis.

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
| `src/aqg/prompts/` | the versioned prompt templates |
| `configs/` | the pinned endpoints (`endpoints.yaml`) and the studies (`study.yaml`, `pilot.yaml`) |
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

## Running a study

A study file in `configs/` names the endpoints, the topics, the number of runs and the seed. Every
topic x endpoint x run is one cell: one call that asks for 5 questions.

```
python -m aqg.run configs/pilot.yaml --plan      # the cells and a rough cost estimate; calls nothing
python -m aqg.run configs/pilot.yaml             # run the study, or resume it
python -m aqg.run configs/pilot.yaml --status    # progress, read from the database
```

- **One contract for every endpoint** (`contract.py`, `prompts/`): the same prompt, settings and seed rules.
  The prompt and the exact messages are hashed.
- **Every reply is checked** (`validate.py`): JSON parsing, the schema, item-writing flags and whether the
  quoted evidence occurs in the reference text. Deviations whose meaning is unambiguous (a code fence, an
  answer given as "B)") are repaired the same way for every endpoint and recorded; the strict results are kept
  for the analysis. A reply that cannot be used is recorded, and the call is repeated, up to 4 attempts.
- **Answer keys are rebalanced** with a seeded shuffle, and the model's own key position is kept.
- **Nothing is lost** (`store.py`): one SQLite file per study holds the settings, the topic texts as sent,
  every call attempt and every question. Rows can only be added, and a study can only be resumed with the
  same settings, prompt and topic texts.
- **Runs are interleaved**: each endpoint works through its cells in a seeded random order, so topics and
  runs are spread over the whole run time. A request or account error stops that endpoint; the budget in the
  study file stops the whole run.

Topic files (`topics.py`) are Markdown files with a YAML header: the official topic, its sub-questions and
the provenance of the reference text.

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

Course materials and the topic reference texts are not part of this repository (copyright); a study names the
folder that holds the topic files. Study databases are written to `outputs/`, which git ignores. The repository
holds the code and the experiment protocol. The generated questions and their scores will be published after the
thesis is assessed.

## License

MIT, see `LICENSE`.
