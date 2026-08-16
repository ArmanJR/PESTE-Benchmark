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

Coordinate schema/registry, adapter, semantic policy, prefetch, carrier Dockerfile/runtime locks, contract
tests, and documentation. Use another runtime when dependency stacks conflict. Never add silent
precision, decoder, batch, or dependency fallback.

Qualification requires mocked multi-item contracts, real offline smoke runs, singleton
equivalence, output-order/cardinality checks, and an x86-64 carrier build using the pinned NGC
digest. The public carrier image holds separate modern and NeMo virtual environments, asserts
distributed support and the expected PyTorch build in both, and includes no host-specific shim.

## Reference Vast.ai session

Configure the key once. Trigger the manual `Runtime image` GitHub Actions workflow for the exact
commit to test, download its `runtime-image.json` artifact, and make the GHCR package public on its
first publication. Then provision an ordinary container by immutable digest:

```bash
uv sync --frozen --all-groups
uv run vastai set api-key <key>
gh workflow run runtime-image.yml --ref <40-character-commit>
# Wait for the run and download its runtime-image-<commit> artifact.
image_ref=$(jq -r .image_reference runtime-image.json)
uv run peste cloud up --image "$image_ref" --max-dph <cap>
```

The search policy preselects verified, reliable `RTX_6000Ada` offers with one GPU, driver
`>=580.0.0`, direct networking, sufficient CPU/RAM/disk, and 300 W capability. Provisioning
allocates 200 GB. `cloud up` tries a bounded number of offers; every rejected or failed instance is
destroyed before the next offer, and only the doctor defines acceptance. Because the carrier is
large, a continuously loading instance gets up to one hour to become SSH-ready.

Large campaigns must total the size of every pinned Hub snapshot because prefetch retains all
repository files. Use `cloud up --disk-gb <size>` to request more than the 200 GB default while
preserving the doctor's 100 GiB free-space floor.

If Vast reports a host-side provisioning failure before SSH or doctor execution, record its offer
ID and exclude it from the next attempt rather than paying to reproduce the same failure:

```bash
uv run peste cloud up --image "$image_ref" --exclude-offer <offer-id>
```

Use the printed direct SSH URL:

```bash
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste model validate --model <model-id> --host <ssh-url>
uv run peste model profile-speed --model <model-id> --host <ssh-url>
```

Commit the calibrated `speed_profile.batch_size` only after reviewing all candidates, the 85%
headroom stress result, singleton conformance, and the 95% knee decision. Because it changes the
model digest, perform calibration for every model before full runs. Destroy the calibration
instance, build the commit containing all reviewed profiles, and provision a fresh container from
that workflow's immutable digest: the carrier image contains the model JSON files, and a stale
image will not match the updated request digest.

```bash
uv run peste validate-specs
uv run peste cloud down
# Trigger and download the final Runtime image workflow artifact.
image_ref=$(jq -r .image_reference runtime-image.json)
uv run peste cloud up --image "$image_ref" --max-dph <cap>
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
```

Run exactly one fresh evaluation per model:

```bash
uv run peste run --suite fleurs-fa-ir-v1 --model <model-id> --host <ssh-url>
```

Resume exists for prediction recovery and accuracy only. A resumed bundle must show
`speed.valid=false`; rerun from scratch for speed publication.

Review status, spec/source/image digests, exact doctor facts, cloud provenance when available,
prediction order/count, journal batch plan, two warmups, timing reciprocity, model facts, WER/CER
aggregates, uncertainty, and structured logs. Confirm the doctor recorded the native socket guard
and unprivileged cache-write probe as enforced.

Finish every session with:

```bash
uv run peste cloud down
```

Destroying the container erases remote state. Result copy-back must be complete first. Stopping is not a
substitute because disk remains billable.

## Multi-model campaigns

A tracked campaign manifest fixes exact candidate identities and ordering. Keep verbose evidence
outside git and use the campaign commands so progress is written atomically and reused only when
the suite, model, source, image, hardware-profile, and GPU identities still match:

```bash
uv run peste campaign qualify \
  --campaign compatible-37-20260811 \
  --host <ssh-url> \
  --evidence-dir campaign-evidence/compatible-37-20260811
uv run peste campaign apply-profiles \
  --campaign compatible-37-20260811 \
  --evidence-dir campaign-evidence/compatible-37-20260811 \
  --remove-failed
uv run peste campaign summarize \
  --campaign compatible-37-20260811 \
  --evidence-dir campaign-evidence/compatible-37-20260811 \
  --output campaigns/compatible-37-20260811/qualification-summary.json
```

After applying profiles, destroy the calibration instance and rebuild from the exact commit with
the reviewed batch sizes. On a fresh accepted host, use `peste campaign run` with the same
evidence directory. It skips an exact successful bundle, never resumes for speed, records each
failure, and continues to the next independent model. Read the complete failure before explicitly
retrying; never change precision, decoding, or batching merely to get a score.

`peste leaderboard --include-untracked` may render reviewed bundles before publication. Final
publication still requires tracked bundles and `peste check-generated`.

### PESTE 2.1.0 Whisper refresh

The tracked `whisper-longform-2-1-0` campaign contains all 29 Whisper specifications affected by
the corrected non-truncating input policy. Use provisional batch size 1 in the calibration image:

```bash
uv run peste campaign qualify \
  --campaign whisper-longform-2-1-0 \
  --host <ssh-url> \
  --evidence-dir campaign-evidence/whisper-longform-2-1-0
uv run peste campaign apply-profiles \
  --campaign whisper-longform-2-1-0 \
  --evidence-dir campaign-evidence/whisper-longform-2-1-0
uv run peste campaign summarize \
  --campaign whisper-longform-2-1-0 \
  --evidence-dir campaign-evidence/whisper-longform-2-1-0 \
  --output campaigns/whisper-longform-2-1-0/qualification-summary.json
```

Rebuild the 2.1 carrier after applying every reviewed profile, provision a fresh accepted host,
prepare the dataset, and run the same campaign. Preserve the old Whisper bundles as stale audit
evidence. Do not remove or regenerate the 13 unchanged non-Whisper 2.0 bundles.

## Publishing

Track one complete successful bundle per model and regenerate all derived outputs together:

```bash
uv run peste leaderboard --suite fleurs-fa-ir-v1
uv run peste check-generated
```

Review `docs/full-leaderboard.md`, accuracy/speed/Pareto SVGs, JSON, CSV, and the README block. Do
not edit derived metrics, standings, or plots directly. Failed attempts may remain as diagnostics
but never rank.

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
