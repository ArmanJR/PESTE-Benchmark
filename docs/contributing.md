# Contributing source changes

Use this guide for benchmark code, tests, runtimes, contracts, documentation, and generated
outputs. Use [Adding a model](adding-a-model.md) for a compatible checkpoint proposal.

## Development setup

PESTE requires Python 3.12 and uses `uv` exclusively:

```bash
uv sync --frozen --all-groups
```

| Path | Responsibility |
|---|---|
| [`src/peste/`](../src/peste) | CLI, cloud, orchestration, adapters, profiling, scoring, generation |
| [`tests/`](../tests) | Unit, batching, timing, doctor, cloud, profiler, adapter contracts |
| [`hardware/`](../hardware) | Immutable doctor-enforced hardware profiles |
| [`models/`](../models) | Checkpoint and deterministic batch specifications |
| [`suites/`](../suites) | Dataset contracts and manifests |
| [`runtimes/`](../runtimes) | Digest-pinned carrier image, offline guard, and frozen environments |
| [`results/`](../results) | Official schema-2 run bundles |
| [`generated/`](../generated) | Deterministic accuracy/speed artifacts |

Read the [benchmark contract](benchmark-contract.md) before changing data, normalization,
batching, timing, hardware acceptance, metrics, resume, or ranking.

Keep changes focused, typed, and structured-logged. Add tests for behavior changes. Never log or
commit API keys, Hub tokens, private values, weights, or audio. Do not add dependencies without a
clear contract need and an updated lockfile. Never weaken determinism, error handling, or tests to
make a failure disappear.

## Validation

```bash
uv run ruff format --check src tests tools
uv run ruff check src tests tools
uv run mypy
uv run pytest
uv run peste validate-specs
uv run peste check-generated
```

If result inputs or rendering change, regenerate with:

```bash
uv run peste leaderboard --suite fleurs-fa-ir-v1
```

That command owns `generated/leaderboard.md`, `generated/leaderboard-accuracy.svg`,
`generated/leaderboard-speed.svg`, `generated/leaderboard-pareto.svg`,
`generated/leaderboard.json`, `generated/leaderboard.csv`, and the README marker block.

Framework/interface changes also require the manual GHCR carrier workflow, a real multi-item smoke,
and speed calibration on the doctor-approved RTX profile. State the changed contract, affected
runtime, image digest, validation performed, and operational follow-up in the pull request. Do not
include unofficial scores.
