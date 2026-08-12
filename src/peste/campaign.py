"""Auditable, resumable orchestration for multi-model benchmark campaigns."""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, cast

from peste.constants import PROJECT_ROOT
from peste.digests import canonical_json, sha256_file
from peste.orchestration import GpuOrchestrator
from peste.schemas import CampaignCandidate, CampaignSpec, RunBundle, RunStatus
from peste.specs import load_model, load_suite, spec_digest
from peste.validation import validate_model_policy

LOGGER = logging.getLogger(__name__)
PROFILE_KNEE_FRACTION = 0.95


def _atomic_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if pretty:
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    else:
        payload = canonical_json(value)
    temporary.write_bytes(payload)
    temporary.replace(path)


def _redact(value: str) -> str:
    redacted = value
    for name in ("HF_TOKEN", "VAST_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _initial_state(campaign: CampaignSpec) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "suite_id": campaign.suite_id,
        "calibration_environment": None,
        "qualification": {},
        "official_environment": None,
        "runs": {},
    }


def _load_state(path: Path, campaign: CampaignSpec) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(campaign)
    try:
        state = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(f"Campaign state is invalid JSON: {path}: {error}") from error
    if state.get("schema_version") != 2:
        raise ValueError(f"Campaign state has unsupported schema version: {path}")
    if state.get("campaign_id") != campaign.campaign_id:
        raise ValueError(f"Campaign state does not match {campaign.campaign_id}: {path}")
    if state.get("suite_id") != campaign.suite_id:
        raise ValueError(f"Campaign state suite does not match {campaign.suite_id}: {path}")
    return state


def _candidate_matches_model(candidate: CampaignCandidate, model: Any) -> None:
    actual = (model.model_id, model.repository, model.revision, model.adapter)
    expected = (
        candidate.model_id,
        candidate.repository,
        candidate.revision,
        candidate.adapter,
    )
    if actual != expected:
        raise ValueError(
            f"Campaign identity does not match model specification for {candidate.model_id}"
        )


def _calibration_identity(
    campaign: CampaignSpec,
    model_digest: str,
    suite_digest: str,
    environment: dict[str, str | int | float | bool],
) -> dict[str, str]:
    return {
        "campaign_id": campaign.campaign_id,
        "suite_digest": suite_digest,
        "model_digest": model_digest,
        "source_revision": str(environment["source_revision"]),
        "image_digest": str(environment["image_digest"]),
        "hardware_profile_id": str(environment["profile_id"]),
        "gpu_uuid": str(environment["gpu_uuid"]),
    }


def validate_profile_result(profile: dict[str, Any]) -> int:
    """Validate profiler evidence and return its deterministic selected batch size."""
    selected = int(profile["selected_batch_size"])
    best = float(profile["best_throughput_x"])
    candidates = profile["candidates"]
    if best <= 0 or not isinstance(candidates, list) or not candidates:
        raise ValueError("Calibration profile has no positive throughput evidence")
    safe = [
        candidate
        for candidate in candidates
        if candidate.get("safe") is True and candidate.get("throughput_x") is not None
    ]
    eligible = sorted(
        int(candidate["batch_size"])
        for candidate in safe
        if float(candidate["throughput_x"]) >= PROFILE_KNEE_FRACTION * best
    )
    if not eligible or selected != eligible[0]:
        raise ValueError("Selected batch size does not match the deterministic 95% knee")
    if not any(
        int(candidate["batch_size"]) == selected and candidate.get("safe") is True
        for candidate in safe
    ):
        raise ValueError("Selected batch size is not a safe candidate")
    return selected


def qualify_campaign(
    campaign: CampaignSpec,
    host: str,
    evidence_directory: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Smoke and calibrate every campaign model, persisting progress after each candidate."""
    state_path = evidence_directory / "state.json"
    state = _load_state(state_path, campaign)
    suite = load_suite(campaign.suite_id, root)
    suite_digest = spec_digest(suite)
    orchestrator = GpuOrchestrator(host, root)
    environment = orchestrator.doctor()
    state["calibration_environment"] = environment
    _atomic_json(state_path, state, pretty=True)

    qualification = state["qualification"]
    for candidate in campaign.candidates:
        model = load_model(candidate.model_id, root)
        _candidate_matches_model(candidate, model)
        validate_model_policy(model, root)
        model_digest = spec_digest(model)
        identity = _calibration_identity(campaign, model_digest, suite_digest, environment)
        previous = qualification.get(candidate.model_id)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "qualified"
            and previous.get("identity") == identity
        ):
            LOGGER.info(
                "Skipping previously qualified campaign model",
                extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
            )
            continue

        LOGGER.info(
            "Qualifying campaign model",
            extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
        )
        try:
            smoke_log = evidence_directory / "smoke" / f"{candidate.model_id}.jsonl"
            profile_log = evidence_directory / "profiles" / f"{candidate.model_id}.jsonl"
            orchestrator.smoke(suite, model, log_path=smoke_log)
            profile = orchestrator.profile_speed(suite, model, log_path=profile_log)
            selected = validate_profile_result(profile)
            profile_path = evidence_directory / "profiles" / f"{candidate.model_id}.json"
            _atomic_json(profile_path, profile, pretty=True)
            qualification[candidate.model_id] = {
                "status": "qualified",
                "identity": identity,
                "selected_batch_size": selected,
                "profile_path": str(profile_path.relative_to(evidence_directory)),
                "profile_sha256": sha256_file(profile_path),
            }
            LOGGER.info(
                "Qualified campaign model",
                extra={
                    "campaign": campaign.campaign_id,
                    "model": candidate.model_id,
                    "selected_batch_size": selected,
                },
            )
        except Exception as error:
            rendered_error = _redact("".join(traceback.format_exception(error)))
            error_path = evidence_directory / "errors" / f"{candidate.model_id}.log"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(rendered_error, encoding="utf-8")
            qualification[candidate.model_id] = {
                "status": "failed",
                "identity": identity,
                "stage": "qualification",
                "error_type": type(error).__name__,
                "error": _redact(str(error)),
                "error_path": str(error_path.relative_to(evidence_directory)),
                "error_sha256": sha256_file(error_path),
            }
            LOGGER.exception(
                "Campaign model qualification failed",
                extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
            )
        _atomic_json(state_path, state, pretty=True)
    return state


def _qualification_summary(
    campaign: CampaignSpec, state: dict[str, Any], root: Path
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    qualification = state["qualification"]
    runs = state.get("runs", {})
    for candidate in campaign.candidates:
        outcome = qualification.get(candidate.model_id, {"status": "pending"})
        summary: dict[str, Any] = {
            "model_id": candidate.model_id,
            "repository": candidate.repository,
            "revision": candidate.revision,
            "adapter": candidate.adapter,
            "qualification_status": outcome["status"],
        }
        for field in (
            "selected_batch_size",
            "profile_sha256",
            "stage",
            "error_type",
            "error",
            "error_sha256",
        ):
            if field in outcome:
                summary[field] = outcome[field]
        model_path = root / "models" / f"{candidate.model_id}.json"
        if model_path.exists():
            summary["final_model_digest"] = spec_digest(load_model(candidate.model_id, root))
        run = runs.get(candidate.model_id)
        if isinstance(run, dict):
            summary["run_status"] = run.get("status")
            if "run_id" in run:
                summary["run_id"] = run["run_id"]
        models.append(summary)
    return {
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "suite_id": campaign.suite_id,
        "candidate_count": len(campaign.candidates),
        "calibration_environment": state.get("calibration_environment"),
        "official_environment": state.get("official_environment"),
        "models": models,
    }


def write_campaign_summary(
    campaign: CampaignSpec,
    evidence_directory: Path,
    output: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    state = _load_state(evidence_directory / "state.json", campaign)
    summary = _qualification_summary(campaign, state, root)
    _atomic_json(output, summary, pretty=True)
    LOGGER.info(
        "Wrote tracked campaign summary",
        extra={"campaign": campaign.campaign_id, "output": str(output)},
    )
    return summary


def apply_campaign_profiles(
    campaign: CampaignSpec,
    evidence_directory: Path,
    *,
    remove_failed: bool,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Apply reviewed calibrated batches and optionally remove failed provisional specs."""
    state = _load_state(evidence_directory / "state.json", campaign)
    qualification = state["qualification"]
    for candidate in campaign.candidates:
        outcome = qualification.get(candidate.model_id)
        if not isinstance(outcome, dict):
            raise ValueError(f"Campaign model has no qualification result: {candidate.model_id}")
        model_path = root / "models" / f"{candidate.model_id}.json"
        if outcome.get("status") != "qualified":
            if remove_failed and model_path.exists():
                model_path.unlink()
                LOGGER.warning(
                    "Removed failed provisional model specification",
                    extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
                )
            continue
        profile_path = evidence_directory / str(outcome["profile_path"])
        if sha256_file(profile_path) != outcome["profile_sha256"]:
            raise ValueError(f"Calibration evidence digest mismatch for {candidate.model_id}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        selected = validate_profile_result(profile)
        model = load_model(candidate.model_id, root)
        _candidate_matches_model(candidate, model)
        updated = model.model_copy(
            update={
                "speed_profile": model.speed_profile.model_copy(update={"batch_size": selected})
            }
        )
        _atomic_json(model_path, updated.model_dump(mode="json"), pretty=True)
        outcome["final_model_digest"] = spec_digest(updated)
        LOGGER.info(
            "Applied calibrated model speed profile",
            extra={
                "campaign": campaign.campaign_id,
                "model": candidate.model_id,
                "selected_batch_size": selected,
            },
        )
    _atomic_json(evidence_directory / "state.json", state, pretty=True)
    return state


def _matching_successful_bundle(root: Path, campaign: CampaignSpec, model: Any) -> RunBundle | None:
    suite = load_suite(campaign.suite_id, root)
    suite_digest = spec_digest(suite)
    model_digest = spec_digest(model)
    results = root / "results" / campaign.suite_id
    matches: list[RunBundle] = []
    for run_path in sorted(results.glob("*/run.json")):
        bundle = RunBundle.model_validate_json(run_path.read_text(encoding="utf-8"))
        if (
            bundle.status == RunStatus.SUCCESS
            and bundle.model_id == model.model_id
            and bundle.suite_digest == suite_digest
            and bundle.model_digest == model_digest
        ):
            matches.append(bundle)
    if len(matches) > 1:
        raise ValueError(f"Multiple successful campaign bundles exist for {model.model_id}")
    return matches[0] if matches else None


def run_campaign(
    campaign: CampaignSpec,
    host: str,
    evidence_directory: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run each qualified campaign model once without speed-invalidating resume."""
    state_path = evidence_directory / "state.json"
    state = _load_state(state_path, campaign)
    suite = load_suite(campaign.suite_id, root)
    orchestrator = GpuOrchestrator(host, root)
    state["official_environment"] = orchestrator.doctor()
    _atomic_json(state_path, state, pretty=True)
    qualification = state["qualification"]
    runs = state["runs"]

    for candidate in sorted(campaign.candidates, key=lambda item: item.model_id):
        outcome = qualification.get(candidate.model_id)
        if not isinstance(outcome, dict) or outcome.get("status") != "qualified":
            LOGGER.info(
                "Skipping campaign model that did not qualify",
                extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
            )
            continue
        model_path = root / "models" / f"{candidate.model_id}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Qualified model specification is missing: {model_path}")
        model = load_model(candidate.model_id, root)
        _candidate_matches_model(candidate, model)
        if model.speed_profile.batch_size != int(outcome["selected_batch_size"]):
            raise ValueError(f"Calibrated batch mismatch for {candidate.model_id}")
        existing = _matching_successful_bundle(root, campaign, model)
        if existing is not None:
            runs[candidate.model_id] = {"status": "success", "run_id": existing.run_id}
            _atomic_json(state_path, state, pretty=True)
            LOGGER.info(
                "Skipping existing successful campaign bundle",
                extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
            )
            continue
        LOGGER.info(
            "Starting official campaign run",
            extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
        )
        try:
            bundle = orchestrator.run(suite, model, resume=False)
            runs[candidate.model_id] = {
                "status": bundle.status.value,
                "run_id": bundle.run_id,
                "speed_valid": bundle.speed.valid,
            }
            if bundle.status != RunStatus.SUCCESS:
                LOGGER.error(
                    "Official campaign model run did not succeed",
                    extra={
                        "campaign": campaign.campaign_id,
                        "model": candidate.model_id,
                        "run": bundle.run_id,
                        "status": bundle.status.value,
                    },
                )
        except Exception as error:
            rendered_error = _redact("".join(traceback.format_exception(error)))
            error_path = evidence_directory / "run-errors" / f"{candidate.model_id}.log"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(rendered_error, encoding="utf-8")
            runs[candidate.model_id] = {
                "status": "orchestration-failed",
                "error_type": type(error).__name__,
                "error": _redact(str(error)),
                "error_sha256": sha256_file(error_path),
            }
            LOGGER.exception(
                "Official campaign orchestration failed",
                extra={"campaign": campaign.campaign_id, "model": candidate.model_id},
            )
        _atomic_json(state_path, state, pretty=True)
    return state
