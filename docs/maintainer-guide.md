# Maintainer guide

This guide covers adapter/runtime changes, speed calibration, official cloud sessions, result
review, publication, and suite creation. Model contributors should use
[Adding a model](adding-a-model.md).

## Adapter and runtime work

An `ASRAdapter` implementation must:

- load the pinned checkpoint at native precision;
- implement `transcribe_batch(audio_paths)` with real framework batching;
- return exactly one ordered transcription per input;
- report parameter count; and
- release resources without hiding failures.

Coordinate schema/registry, adapter, semantic policy, prefetch, runtime Dockerfile/lock, contract
tests, and documentation. Use another runtime when dependency stacks conflict. Never add silent
precision, decoder, batch, or dependency fallback.

Qualification requires mocked multi-item contracts, real offline smoke runs, singleton
equivalence, output-order/cardinality checks, and an x86-64 runtime build using the pinned NGC
digest. Runtime image builds assert distributed support and the expected PyTorch build, so no
host-specific compatibility shim is allowed.

## Reference Vast.ai session

Configure the key once, then provision a VM:

```bash
uv sync --frozen --all-groups
uv run vastai set api-key <key>
uv run peste cloud up --max-dph <cap>
uv run peste cloud build
```

The search policy preselects verified, reliable `RTX_6000Ada` offers with one GPU, VM support,
direct networking, sufficient CPU/RAM/disk, 300 W capability, and the pinned driver. `cloud up`
tries a bounded number of offers. Every rejected or failed instance is destroyed before the next
offer; only the doctor defines acceptance.

Use the printed direct SSH URL:

```bash
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste model validate --model <model-id> --host <ssh-url>
uv run peste model profile-speed --model <model-id> --host <ssh-url>
```

Commit the calibrated `speed_profile.batch_size` only after reviewing all candidates, the 85%
headroom stress result, singleton conformance, and the 95% knee decision. Because it changes the
model digest, perform calibration for every model before full runs. Rebuild both runtime images
after committing the profiles: the images contain the model JSON files, and a stale image will not
match the updated request digest.

```bash
uv run peste validate-specs
uv run peste cloud build
```

Run exactly one fresh evaluation per model:

```bash
uv run peste run --suite fleurs-fa-ir-v1 --model <model-id> --host <ssh-url>
```

Resume exists for prediction recovery and accuracy only. A resumed bundle must show
`speed.valid=false`; rerun from scratch for speed publication.

Review status, spec/source/image digests, exact doctor facts, cloud provenance when available,
prediction order/count, journal batch plan, two warmups, timing reciprocity, model facts, WER/CER
aggregates, uncertainty, and structured logs. Confirm networking was disabled and caches read-only
during inference.

Finish every session with:

```bash
uv run peste cloud down
```

Destroying the VM erases remote state. Result copy-back must be complete first. Stopping is not a
substitute because disk remains billable.

## Publishing

Track one complete successful bundle per model and regenerate all derived outputs together:

```bash
uv run peste leaderboard --suite fleurs-fa-ir-v1
uv run peste check-generated
```

Review Markdown, accuracy/speed SVGs, JSON, CSV, and the README block. Do not edit derived metrics
or plots directly. Failed attempts may remain as diagnostics but never rank.

## Creating a suite

A source, transcript, audio, split, or normalization change requires a new immutable suite ID and
independent results. Pin provenance/license, define split counts, materialize canonical audio,
reject empty normalized references, review the complete manifest, and store its SHA-256. Use
[`tools/freeze_suite.py`](../tools/freeze_suite.py) only for initial creation; it refuses overwrite.

## Required validation

```bash
uv run ruff format --check src tests tools
uv run ruff check src tests tools
uv run mypy
uv run pytest
uv run peste validate-specs
uv run peste check-generated
```
