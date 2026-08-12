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

Qualification completed on 2026-08-12 from the digest-pinned calibration carrier. Thirty-four
models passed pinned prefetch, real offline smoke, multi-item singleton equivalence,
cardinality/order checks, 85% VRAM headroom, and deterministic 95%-knee selection. Their selected
batches are 27 at batch 1, four at batch 2, one at batch 4, one at batch 32, and one at batch 128.
The three Zoha checkpoints failed because their processors omitted the attention mask required for
padding-safe CTC decoding; their provisional specifications were removed.

The tracked `qualification-summary.json` preserves compact outcomes and evidence hashes. Verbose
smoke, calibration, and failure logs remain under ignored `campaign-evidence/`. Calibration
throughput is evidence, not an unofficial benchmark score.

The two Shenava candidates remain eligible under the unchanged benchmark contract, but their
published FLEURS training/model-selection overlap must remain disclosed when results are
published. No external language model, alternate decoder, precision fallback, or model-specific
repair is permitted.
