# Contributing source changes

This guide covers changes to benchmark code, tests, runtimes, documentation, validation, metrics,
and generated outputs. To propose a checkpoint supported by an existing adapter, use
[Adding a model](adding-a-model.md) instead.

## Development setup

PESTE requires Python 3.12 and uses `uv` for dependency and command execution.

```bash
uv sync --frozen --all-groups
```

The main source areas are:

| Path | Responsibility |
|---|---|
| [`src/peste/`](../src/peste) | CLI, orchestration, adapters, scoring, validation, and generation |
| [`tests/`](../tests) | Unit, runner, smoke, and adapter-contract tests |
| [`models/`](../models) | Immutable checkpoint specifications |
| [`suites/`](../suites) | Immutable dataset specifications and manifests |
| [`runtimes/`](../runtimes) | Framework-specific images and frozen dependency sets |
| [`results/`](../results) | Official run bundles and diagnostics |
| [`generated/`](../generated) | Deterministic leaderboard artifacts |

Read the [benchmark contract](benchmark-contract.md) before changing behavior that affects data,
normalization, inference, scoring, result eligibility, or ranking.

## Change requirements

- Keep each pull request focused on one behavior or closely related set of changes.
- Follow existing architecture, naming, typing, and structured-logging patterns.
- Use standard logging rather than `print`; retain context and full tracebacks for failures.
- Add or update tests for behavior changes. Do not weaken tests to accommodate an unintended
  regression.
- Handle validation and error cases explicitly; do not silently fall back to another policy.
- Do not log, hardcode, or commit credentials, private environment values, model weights, or
  dataset audio.
- Reuse existing dependencies where practical. Runtime dependency changes must update the
  corresponding `pyproject.toml` and frozen `uv.lock` together.
- Do not edit generated, vendored, compiled, minified, or lock-generated content when an
  authoritative source or generation command exists.

Changes to an immutable published suite, model policy, or official result require maintainer
review. Do not rewrite published manifests or result bundles.

## Validation

Run the complete local validation set before opening a pull request:

```bash
uv run --frozen ruff format --check src tests tools runtimes/nemo/compat runtimes/transformers-compat
uv run --frozen ruff check src tests tools runtimes/nemo/compat runtimes/transformers-compat
uv run --frozen mypy
uv run --frozen pytest
uv run --frozen peste validate-specs
uv run --frozen peste check-generated
```

Use the formatter without `--check` on files you changed when formatting is required. Keep
unrelated formatting and refactors out of the pull request.

If leaderboard rendering or official result inputs change, regenerate all derived outputs with:

```bash
uv run --frozen peste leaderboard --suite fleurs-fa-ir-v1
```

This command owns `generated/leaderboard.md`, `generated/leaderboard-accuracy.svg`,
`generated/leaderboard-memory.svg`, `generated/leaderboard.json`, `generated/leaderboard.csv`,
and the leaderboard block in the README. Do not edit those generated sections directly.

## Runtime and adapter changes

Framework or inference-interface changes require more than ordinary unit coverage. They must
include mocked adapter-contract tests and pass a real one-sample deterministic smoke test on the
official Jetson before a full evaluation is accepted. These changes are coordinated by benchmark
maintainers; see the [maintainer guide](maintainer-guide.md).

## Pull-request content

The pull-request description should state:

- the problem and intended behavior;
- the benchmark contract, if any, that changes;
- files and runtime families affected;
- validation performed and its result; and
- known limitations or follow-up work.

Do not include unofficial benchmark scores. Official result bundles are produced and published
through the maintainer workflow.
