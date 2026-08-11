"""PESTE command-line interface."""

import filecmp
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from peste.constants import DEFAULT_SEED, DEFAULT_SUITE_ID, PROJECT_ROOT
from peste.dataset import prepare_dataset
from peste.digests import canonical_json
from peste.leaderboard import generate_leaderboards
from peste.logging import configure_logging
from peste.manifest import validate_manifest
from peste.prefetch import prefetch_model
from peste.runner import run_benchmark
from peste.schemas import RunRequest
from peste.smoke import smoke_adapter
from peste.specs import discover_models, load_model, load_suite
from peste.validation import validate_model_policy

LOGGER = logging.getLogger(__name__)
app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="PESTE v2 schema-2 Persian ASR accuracy and batched-speed benchmark.",
)
dataset_app = typer.Typer(no_args_is_help=True, help="Prepare immutable suite audio on a GPU VM.")
model_app = typer.Typer(
    no_args_is_help=True,
    help="Validate adapters and calibrate deterministic RTX speed-profile batch sizes.",
)
cloud_app = typer.Typer(
    no_args_is_help=True,
    help="Provision, inspect, build on, and destroy doctor-gated Vast.ai VM instances.",
)
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(cloud_app, name="cloud")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("peste-benchmark"))
        raise typer.Exit()


@app.callback()
def main(
    log_level: str = typer.Option("INFO", envvar="PESTE_LOG_LEVEL"),
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    del version
    configure_logging(log_level)


@app.command(help="Enforce the rtx-6000-ada-v1 hardware contract over Docker SSH.")
def doctor(
    host: str = typer.Option(..., help="Docker endpoint, for example ssh://root@host:22"),
) -> None:
    from peste.orchestration import GpuOrchestrator

    report = GpuOrchestrator(host).doctor()
    LOGGER.info("Doctor report", extra={"host": host, "profile": report})


@dataset_app.command("prepare")
def dataset_prepare(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    host: str = typer.Option(..., help="Doctor-approved Docker-over-SSH endpoint"),
) -> None:
    from peste.orchestration import GpuOrchestrator

    spec = load_suite(suite)
    GpuOrchestrator(host).prepare_dataset(spec)
    LOGGER.info("Remote dataset is prepared", extra={"suite": suite, "host": host})


@model_app.command("validate")
def model_validate(
    model: str = typer.Option(...),
    host: str | None = typer.Option(None, help="Also run a real offline RTX smoke test"),
) -> None:
    spec = load_model(model)
    validate_model_policy(spec, PROJECT_ROOT)
    LOGGER.info("Model specification is valid", extra={"model": model})
    if host is not None:
        from peste.orchestration import GpuOrchestrator

        GpuOrchestrator(host).smoke(load_suite(DEFAULT_SUITE_ID), spec)


@model_app.command(
    "profile-speed",
    help="Calibrate candidates 1..128 and report the deterministic 95%-knee batch size.",
)
def model_profile_speed(
    model: str = typer.Option(...),
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    host: str = typer.Option(..., help="Doctor-approved Docker-over-SSH endpoint"),
) -> None:
    from peste.orchestration import GpuOrchestrator

    result = GpuOrchestrator(host).profile_speed(load_suite(suite), load_model(model))
    LOGGER.info("Speed profile calibration complete", extra={"model": model, **result})
    typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))


@app.command("run", help="Run one model with its committed batch size; resumed speed is invalid.")
def run_command(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    model: str = typer.Option(...),
    host: str = typer.Option(...),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    from peste.orchestration import GpuOrchestrator

    suite_spec = load_suite(suite)
    model_spec = load_model(model)
    bundle = GpuOrchestrator(host).run(suite_spec, model_spec, resume=resume)
    LOGGER.info(
        "Remote run finished",
        extra={"run": bundle.run_id, "model": model, "status": bundle.status.value},
    )


@app.command("run-all", help="Run every model with committed deterministic speed profiles.")
def run_all(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    host: str = typer.Option(...),
) -> None:
    from peste.orchestration import GpuOrchestrator

    suite_spec = load_suite(suite)
    orchestrator = GpuOrchestrator(host)
    for model_spec in discover_models():
        bundle = orchestrator.run(suite_spec, model_spec)
        LOGGER.info(
            "Model run finalized",
            extra={
                "run": bundle.run_id,
                "model": model_spec.model_id,
                "status": bundle.status.value,
            },
        )


@app.command()
def leaderboard(suite: str = typer.Option(DEFAULT_SUITE_ID)) -> None:
    suite_spec = load_suite(suite)
    generate_leaderboards(
        suite_spec,
        PROJECT_ROOT / "results" / suite,
        PROJECT_ROOT / "generated",
    )


@app.command("validate-specs", hidden=True)
def validate_specs() -> None:
    suite = load_suite(DEFAULT_SUITE_ID)
    validate_manifest(suite, PROJECT_ROOT / "suites" / suite.suite_id)
    for model in discover_models():
        validate_model_policy(model, PROJECT_ROOT)
    LOGGER.info("All suite and model specifications are valid")


@app.command("check-generated", hidden=True)
def check_generated() -> None:
    suite = load_suite(DEFAULT_SUITE_ID)
    tracked_files = (
        "leaderboard.md",
        "leaderboard.json",
        "leaderboard.csv",
        "leaderboard-accuracy.svg",
        "leaderboard-speed.svg",
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        temporary_generated = temporary_root / "generated"
        temporary_readme = temporary_root / "README.md"
        shutil.copytree(PROJECT_ROOT / "models", temporary_root / "models")
        temporary_readme.write_text(
            (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        generate_leaderboards(
            suite,
            PROJECT_ROOT / "results" / suite.suite_id,
            temporary_generated,
            root=temporary_root,
            require_tracked=False,
        )
        for name in tracked_files:
            if not filecmp.cmp(
                PROJECT_ROOT / "generated" / name, temporary_generated / name, shallow=False
            ):
                raise RuntimeError(f"Generated output is stale: generated/{name}")
        if not filecmp.cmp(PROJECT_ROOT / "README.md", temporary_readme, shallow=False):
            raise RuntimeError("Generated README leaderboard is stale")
    LOGGER.info("Generated leaderboard outputs are current")


@app.command("_dataset-container", hidden=True)
def dataset_container(suite: str = typer.Option(DEFAULT_SUITE_ID)) -> None:
    spec = load_suite(suite)
    prepare_dataset(spec, PROJECT_ROOT / "suites" / suite, Path("/cache/dataset"))


@app.command("_prefetch-container", hidden=True)
def prefetch_container(model: str = typer.Option(...)) -> None:
    prefetch_model(load_model(model), Path("/cache/hf"))


@app.command("_run-container", hidden=True)
def run_container() -> None:
    raw_request = os.environ.get("PESTE_RUN_REQUEST_JSON")
    if raw_request is None:
        raise RuntimeError("PESTE_RUN_REQUEST_JSON is required")
    request = RunRequest.model_validate_json(raw_request)
    suite = load_suite(request.suite_id)
    model = load_model(request.model_id)
    bundle = run_benchmark(
        request,
        suite,
        model,
        PROJECT_ROOT / "suites" / suite.suite_id,
    )
    if bundle.status.value != "success":
        raise typer.Exit(code=1)


@app.command("_smoke-container", hidden=True)
def smoke_container(
    suite: str = typer.Option(DEFAULT_SUITE_ID), model: str = typer.Option(...)
) -> None:
    suite_spec = load_suite(suite)
    smoke_adapter(
        suite_spec,
        load_model(model),
        PROJECT_ROOT / "suites" / suite,
        Path("/cache/dataset"),
        Path("/cache/hf"),
        seed=DEFAULT_SEED,
    )


@app.command("_profile-speed-container", hidden=True)
def profile_speed_container(
    output: Annotated[Path, typer.Option()],
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    model: str = typer.Option(...),
) -> None:
    from peste.adapters import create_adapter
    from peste.profiling import calibrate_batch_size
    from peste.runner import _seed_runtime

    suite_spec = load_suite(suite)
    model_spec = load_model(model)
    rows = [
        row
        for row in validate_manifest(suite_spec, PROJECT_ROOT / "suites" / suite_spec.suite_id)
        if row.split == suite_spec.evaluation_split
    ]
    torch = _seed_runtime(DEFAULT_SEED)
    adapter = create_adapter(model_spec, Path("/cache/hf"))
    try:
        adapter.load()
        result = calibrate_batch_size(
            adapter,
            rows,
            Path("/cache/dataset"),
            suite_spec.normalization_version,
            torch,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json(result.as_dict()))
    finally:
        adapter.close()


@cloud_app.command("up", help="Rent and bootstrap a bounded, doctor-approved Vast.ai VM.")
def cloud_up(
    max_dph: float | None = typer.Option(None, min=0),
    maximum_attempts: int = typer.Option(3, min=1, max=10, hidden=True),
) -> None:
    from peste.cloud import VastClient, provision_official_vm
    from peste.orchestration import GpuOrchestrator

    provisioned = provision_official_vm(
        VastClient(),
        GpuOrchestrator,
        max_dph=max_dph,
        maximum_attempts=maximum_attempts,
    )
    LOGGER.info(
        "Provisioned doctor-approved Vast.ai VM",
        extra={
            "instance_id": provisioned.instance_id,
            "offer_id": provisioned.offer_id,
            "dph_total": provisioned.dph_total,
            "host": provisioned.host,
        },
    )
    typer.echo(f"${provisioned.dph_total:.4f}/hr")
    typer.echo(provisioned.host)


@cloud_app.command("status", help="List peste-official instances, state, and hourly price.")
def cloud_status() -> None:
    from peste.cloud import VastClient, labeled_instances

    instances = labeled_instances(VastClient())
    if not instances:
        typer.echo("No peste-official instances")
        return
    for instance in instances:
        instance_id = instance.get("id", instance.get("instance_id"))
        state = instance.get("actual_status", instance.get("status", "unknown"))
        price = float(instance.get("dph_total", instance.get("dph_base", 0)))
        typer.echo(f"{instance_id}\t{state}\t${price:.4f}/hr")


@cloud_app.command("down", help="Destroy every labeled instance so storage billing also ends.")
def cloud_down() -> None:
    from peste.cloud import VastClient, labeled_instances

    client = VastClient()
    instances = labeled_instances(client)
    failures: list[str] = []
    destroyed = 0
    for instance in instances:
        raw_id = instance.get("id", instance.get("instance_id", instance.get("contract_id")))
        if raw_id is None:
            failures.append(f"missing id: {instance}")
            continue
        instance_id = int(raw_id)
        try:
            client.destroy_instance(instance_id)
            destroyed += 1
        except RuntimeError as error:
            failures.append(f"{instance_id}: {error}")
            LOGGER.exception(
                "Failed to destroy labeled Vast.ai instance",
                extra={"instance_id": instance_id},
            )
    LOGGER.info("Destroyed labeled Vast.ai instances", extra={"count": destroyed})
    if failures:
        raise RuntimeError(
            "Some labeled Vast.ai instances were not destroyed: " + "; ".join(failures)
        )


@cloud_app.command("build", help="Build both v2 runtime images through the VM Docker daemon.")
def cloud_build() -> None:
    from peste.cloud import VastClient, labeled_instances
    from peste.orchestration import GpuOrchestrator

    client = VastClient()
    running = [
        instance
        for instance in labeled_instances(client)
        if instance.get("actual_status", instance.get("status")) == "running"
    ]
    if len(running) != 1:
        raise RuntimeError(f"Expected one running peste-official instance; found {len(running)}")
    host = client.ssh_url(running[0])
    orchestrator = GpuOrchestrator(host)
    orchestrator.doctor()
    orchestrator.build_images()


if __name__ == "__main__":
    app()
