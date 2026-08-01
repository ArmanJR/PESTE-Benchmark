# Maintainer guide

This guide covers administrator-owned work: supporting checkpoints outside existing adapter
contracts, changing runtime stacks, executing official evaluations, publishing results, and
creating benchmark suites. Model contributors should use [Adding a model](adding-a-model.md).

## Supporting an unsupported model

Before implementing support, decide whether the checkpoint needs:

1. a small extension to an existing adapter without changing benchmark policy;
2. a new adapter in an existing runtime; or
3. a new runtime and adapter.

Do not route a checkpoint through an adapter merely because it uses the same framework. Loading,
audio preparation, prompting, decoding, output extraction, precision, and auxiliary artifacts must
all satisfy a documented deterministic contract.

An adapter implementation must provide the interface in
[`ASRAdapter`](../src/peste/adapters/base.py):

- `load()` loads the exact pinned checkpoint at benchmark precision;
- `transcribe()` accepts one canonical WAV and returns scoreable text plus optional structured
  output;
- `parameter_count` reports loaded checkpoint parameters; and
- `close()` releases resources without hiding failures.

## Adapter integration checklist

Adding an adapter normally requires coordinated changes to:

- [`src/peste/schemas.py`](../src/peste/schemas.py) for allowed adapter/runtime identifiers;
- [`src/peste/adapters/`](../src/peste/adapters) for the implementation;
- [`src/peste/adapters/__init__.py`](../src/peste/adapters/__init__.py) for registration;
- [`src/peste/validation.py`](../src/peste/validation.py) for fixed semantic policy;
- [`src/peste/prefetch.py`](../src/peste/prefetch.py) for pinned auxiliary artifacts;
- [`runtimes/`](../runtimes) for the Dockerfile, dependency manifest, and frozen lock;
- [`tests/test_adapter_contracts.py`](../tests/test_adapter_contracts.py) for mocked loading and
  inference contracts; and
- documentation describing compatibility and policy.

Use a separate runtime when dependency requirements conflict with existing frozen stacks or when
framework initialization requires isolated compatibility work. Runtime images must pin their base
image and dependencies and must install PESTE from the evaluated source tree.

Never add silent fallback behavior. Unsupported precision, missing artifacts, incompatible APIs,
nondeterminism, and OOM conditions must fail with structured diagnostics.

## Adapter qualification

Before a complete evaluation:

1. validate schema and semantic policy;
2. run mocked contract tests for loading arguments, device, dtype, generation, and output
   extraction;
3. build the ARM64 Jetson runtime image from the intended source revision;
4. prefetch all checkpoint and auxiliary artifacts at immutable revisions;
5. run the real one-sample smoke test twice and require identical normalized output; and
6. inspect parameter count and peak memory reported by the smoke test.

A smoke test establishes basic compatibility and determinism; it is not a benchmark score.

## Official model evaluation

Run official evaluations from a source revision containing the accepted model specification and
adapter/runtime implementation.

```bash
uv run --frozen peste doctor --host ssh://jetson
uv run --frozen peste dataset prepare --suite fleurs-fa-ir-v1 --host ssh://jetson
uv run --frozen peste model validate --model <model-id>
uv run --frozen peste model validate --model <model-id> --host ssh://jetson
uv run --frozen peste run --suite fleurs-fa-ir-v1 --model <model-id> --host ssh://jetson
```

Review the completed bundle before publication:

- status and complete sample count;
- suite/model digests and source revision;
- runtime image digest and hardware profile;
- dependency, CUDA, and PyTorch fingerprints;
- prediction sequence and sample IDs;
- structured output retention where applicable;
- WER/CER aggregates and edit totals;
- deterministic WER/CER intervals and paired adjacent-model CER comparisons;
- checkpoint bytes and parameter count;
- peak CUDA reserved/allocated memory and process RSS; and
- runner/container logs for warnings or fallbacks.

Do not publish a score from a contributor's hardware or from an altered decoding/precision policy.

## Publishing results

The generator ignores a completely untracked `run.json`. After review, track the complete intended
result bundle and regenerate the outputs:

```bash
uv run --frozen peste leaderboard --suite fleurs-fa-ir-v1
uv run --frozen peste check-generated
```

Review changes to the Markdown tables, SVG plot, JSON, CSV, and README block together. Publish the
bundle and generated artifacts as one coherent result update. Do not edit derived scores or chart
values manually.

Only one successful official bundle may exist for a model ID within a suite. Do not overwrite a
published bundle. A checkpoint revision or benchmark-policy change requires an explicit immutable
identity/version decision before another result is published.

OOM and other failed official attempts may be retained with diagnostics, but they remain unranked.
Do not change precision, decoder, or runtime policy after observing a failure merely to obtain a
ranked score.

## Creating a suite

A new dataset or material change to an existing suite requires a new immutable suite ID and
independent leaderboard. Before sealing a suite:

- document source provenance and license;
- pin every source revision;
- define split counts and the evaluation split;
- select and version the normalization policy;
- materialize canonical audio;
- reject empty normalized references;
- generate and review the complete manifest; and
- record the manifest SHA-256 in the suite specification.

Use [`tools/freeze_suite.py`](../tools/freeze_suite.py) only for initial manifest creation. It
refuses to overwrite an existing manifest. Published suite specifications, manifests, and results
are immutable.

## Required validation

Run the complete CI-equivalent validation before publishing source or result changes:

```bash
uv run --frozen ruff format --check src tests tools runtimes/nemo/compat runtimes/transformers-compat
uv run --frozen ruff check src tests tools runtimes/nemo/compat runtimes/transformers-compat
uv run --frozen mypy
uv run --frozen pytest
uv run --frozen peste validate-specs
uv run --frozen peste check-generated
```
