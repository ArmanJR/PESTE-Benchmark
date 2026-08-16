"""Campaign manifest, evidence, and failure-isolation tests."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from peste import campaign as campaign_module
from peste.constants import PROJECT_ROOT
from peste.digests import sha256_file
from peste.schemas import CampaignSpec
from peste.specs import load_campaign, load_model


def _profile(selected: int = 2) -> dict[str, Any]:
    return {
        "selected_batch_size": selected,
        "best_throughput_x": 10.0,
        "candidates": [
            {
                "batch_size": 1,
                "safe": True,
                "throughput_x": 8.0,
                "peak_vram_fraction": 0.2,
                "rejection_reason": None,
            },
            {
                "batch_size": 2,
                "safe": True,
                "throughput_x": 9.5,
                "peak_vram_fraction": 0.3,
                "rejection_reason": None,
            },
            {
                "batch_size": 4,
                "safe": True,
                "throughput_x": 10.0,
                "peak_vram_fraction": 0.4,
                "rejection_reason": None,
            },
        ],
    }


def _doctor() -> dict[str, str | int | float | bool]:
    return {
        "source_revision": "a" * 40,
        "image_digest": f"sha256:{'b' * 64}",
        "profile_id": "rtx-6000-ada-v1",
        "gpu_uuid": "GPU-test",
    }


def test_compatible_campaign_contains_exact_new_adapter_split() -> None:
    campaign = load_campaign("compatible-37-20260811")
    assert len(campaign.candidates) == 37
    assert Counter(candidate.adapter for candidate in campaign.candidates) == {
        "transformers-whisper": 25,
        "transformers-ctc": 10,
        "nemo-rnnt": 1,
        "nemo-ctc": 1,
    }
    existing = {
        "whisper-large-persian-steja",
        "whisper-persian-paulwalker",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        "qwen3-asr-0-6b",
        "qwen3-asr-1-7b",
        "wav2vec2-large-xlsr-53-persian",
        "nvidia-fastconformer-fa",
    }
    assert not existing.intersection(candidate.model_id for candidate in campaign.candidates)


def test_whisper_longform_campaign_contains_every_whisper_model() -> None:
    campaign = load_campaign("whisper-longform-2-1-0")
    expected = {
        path.stem
        for path in (PROJECT_ROOT / "models").glob("*.json")
        if load_model(path.stem).adapter == "transformers-whisper"
    }

    assert len(campaign.candidates) == 29
    assert {candidate.model_id for candidate in campaign.candidates} == expected
    assert {candidate.adapter for candidate in campaign.candidates} == {"transformers-whisper"}
    failed = {
        "wav2vec2-base-common-voice-persian-colab-zoha",
        "wav2vec2-base-common-voice-40p-persian-colab-zoha",
        "wav2vec2-xlsr-persian-50p-zoha",
    }
    for candidate in campaign.candidates:
        model_path = PROJECT_ROOT / "models" / f"{candidate.model_id}.json"
        if candidate.model_id in failed:
            assert not model_path.exists()
            continue
        model = json.loads(model_path.read_text(encoding="utf-8"))
        assert (model["repository"], model["revision"], model["adapter"]) == (
            candidate.repository,
            candidate.revision,
            candidate.adapter,
        )


def test_profile_validation_requires_smallest_safe_95_percent_knee() -> None:
    assert campaign_module.validate_profile_result(_profile()) == 2
    with pytest.raises(ValueError, match="95% knee"):
        campaign_module.validate_profile_result(_profile(selected=4))


def test_qualification_records_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    full = load_campaign("compatible-37-20260811")
    selected = (full.candidates[0], full.candidates[1])
    campaign = CampaignSpec(
        schema_version=2,
        campaign_id="two-model-test",
        suite_id=full.suite_id,
        candidates=selected,
    )

    class Orchestrator:
        def __init__(self, host: str, root: Path) -> None:
            assert host == "ssh://root@test:22"
            assert root == PROJECT_ROOT

        def doctor(self) -> dict[str, str | int | float | bool]:
            return _doctor()

        def smoke(self, suite: Any, model: Any, *, log_path: Path | None = None) -> None:
            if model.model_id == selected[0].model_id:
                raise RuntimeError("complete smoke failure")
            assert log_path is not None
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text('{"status":"passed"}\n', encoding="utf-8")

        def profile_speed(
            self, suite: Any, model: Any, *, log_path: Path | None = None
        ) -> dict[str, Any]:
            assert model.model_id == selected[1].model_id
            return _profile()

    monkeypatch.setattr(campaign_module, "GpuOrchestrator", Orchestrator)
    state = campaign_module.qualify_campaign(
        campaign,
        "ssh://root@test:22",
        tmp_path,
    )

    assert state["qualification"][selected[0].model_id]["status"] == "failed"
    assert state["qualification"][selected[1].model_id]["status"] == "qualified"
    error_path = tmp_path / state["qualification"][selected[0].model_id]["error_path"]
    assert "complete smoke failure" in error_path.read_text(encoding="utf-8")
    summary = campaign_module._qualification_summary(campaign, state, PROJECT_ROOT)
    failed = next(item for item in summary["models"] if item["model_id"] == selected[0].model_id)
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_sha256"] == state["qualification"][selected[0].model_id]["error_sha256"]
    assert "error" not in failed


def test_campaign_rejects_duplicate_model_identity() -> None:
    candidate = load_campaign("compatible-37-20260811").candidates[0]
    with pytest.raises(ValueError, match="model IDs must be unique"):
        CampaignSpec(
            schema_version=2,
            campaign_id="duplicate-test",
            suite_id="fleurs-fa-ir-v1",
            candidates=(candidate, candidate),
        )


def test_apply_profiles_verifies_evidence_and_updates_only_calibrated_batch(
    tmp_path: Path,
) -> None:
    full = load_campaign("compatible-37-20260811")
    candidate = full.candidates[0]
    campaign = CampaignSpec(
        schema_version=2,
        campaign_id="apply-profile-test",
        suite_id=full.suite_id,
        candidates=(candidate,),
    )
    root = tmp_path / "root"
    models = root / "models"
    models.mkdir(parents=True)
    shutil.copy2(
        PROJECT_ROOT / "models" / f"{candidate.model_id}.json",
        models / f"{candidate.model_id}.json",
    )
    evidence = tmp_path / "evidence"
    profile_path = evidence / "profiles" / f"{candidate.model_id}.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    state = {
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "suite_id": campaign.suite_id,
        "calibration_environment": _doctor(),
        "qualification": {
            candidate.model_id: {
                "status": "qualified",
                "selected_batch_size": 2,
                "profile_path": str(profile_path.relative_to(evidence)),
                "profile_sha256": sha256_file(profile_path),
            }
        },
        "official_environment": None,
        "runs": {},
    }
    (evidence / "state.json").write_text(json.dumps(state), encoding="utf-8")

    campaign_module.apply_campaign_profiles(
        campaign,
        evidence,
        remove_failed=False,
        root=root,
    )

    updated = json.loads((models / f"{candidate.model_id}.json").read_text(encoding="utf-8"))
    assert updated["speed_profile"]["batch_size"] == 2
    assert updated["revision"] == candidate.revision
