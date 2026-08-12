# Compatible 37 campaign

This campaign covers exactly the 37 additional model-only candidates classified as compatible in
[`asr-model-compatibility.md`](../../asr-model-compatibility.md): 25 Transformers Whisper, 10
Transformers CTC, and 2 NeMo RNNT checkpoints. The eight already-published models and every
conditional or incompatible repository are excluded.

The pinned Hugging Face metadata was rechecked on 2026-08-12. All 37 revisions remained public,
ungated, immutable, and licensed as recorded. Generic snapshot prefetch would retain
127,079,029,213 bytes (118.35 GiB); `steja/whisper-small-persian` alone accounts for
29,984,186,828 bytes because its repository contains many files. Calibration and official
sessions therefore request 400 GB rather than the normal 200 GB allocation so the doctor can
continue to enforce its 100 GiB free-space reserve.

The batch size 1 values in the initial candidate specifications are provisional inputs for the
calibration carrier only. A model becomes eligible for the final carrier after pinned prefetch,
two-pass offline smoke, multi-item singleton equivalence, cardinality/order checks, 85% VRAM
headroom, and deterministic 95%-knee selection. Calibration throughput is evidence, not an
unofficial benchmark score, and is intentionally kept out of git.

The two Shenava candidates remain eligible under the unchanged benchmark contract, but their
published FLEURS training/model-selection overlap must remain disclosed when results are
published. No external language model, alternate decoder, precision fallback, or model-specific
repair is permitted.
