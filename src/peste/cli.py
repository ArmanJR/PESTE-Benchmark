"""PESTE command-line interface."""

import filecmp
import logging
import os
import shutil
import tempfile
from pathlib import Path

import typer

from peste.constants import DEFAULT_SEED, DEFAULT_SUITE_ID, PROJECT_ROOT
from peste.dataset import prepare_dataset
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
app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
dataset_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")


@app.callback()
def main(log_level: str = typer.Option("INFO", envvar="PESTE_LOG_LEVEL")) -> None:
    configure_logging(log_level)


@app.command()
def doctor(host: str = typer.Option(..., help="Docker endpoint, for example ssh://jetson")) -> None:
    from peste.orchestration import JetsonOrchestrator

    report = JetsonOrchestrator(host).doctor()
    LOGGER.info("Doctor report", extra={"host": host, "profile": report})


@dataset_app.command("prepare")
def dataset_prepare(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    host: str = typer.Option(..., help="Docker endpoint, for example ssh://jetson"),
) -> None:
    from peste.orchestration import JetsonOrchestrator

    spec = load_suite(suite)
    JetsonOrchestrator(host).prepare_dataset(spec)
    LOGGER.info("Remote dataset is prepared", extra={"suite": suite, "host": host})


@model_app.command("validate")
def model_validate(
    model: str = typer.Option(...),
    host: str | None = typer.Option(None, help="Also run a real offline Jetson smoke test"),
) -> None:
    spec = load_model(model)
    validate_model_policy(spec, PROJECT_ROOT)
    LOGGER.info("Model specification is valid", extra={"model": model})
    if host is not None:
        from peste.orchestration import JetsonOrchestrator

        JetsonOrchestrator(host).smoke(load_suite(DEFAULT_SUITE_ID), spec)


@app.command("run")
def run_command(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    model: str = typer.Option(...),
    host: str = typer.Option(...),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    from peste.orchestration import JetsonOrchestrator

    suite_spec = load_suite(suite)
    model_spec = load_model(model)
    bundle = JetsonOrchestrator(host).run(suite_spec, model_spec, resume=resume)
    LOGGER.info(
        "Remote run finished",
        extra={"run": bundle.run_id, "model": model, "status": bundle.status.value},
    )


@app.command("run-all")
def run_all(
    suite: str = typer.Option(DEFAULT_SUITE_ID),
    host: str = typer.Option(...),
) -> None:
    from peste.orchestration import JetsonOrchestrator

    suite_spec = load_suite(suite)
    orchestrator = JetsonOrchestrator(host)
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
        "leaderboard-memory.svg",
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


if __name__ == "__main__":
    app()
