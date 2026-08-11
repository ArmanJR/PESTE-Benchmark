"""Public CLI surface tests."""

from typer.testing import CliRunner

from peste.cli import app


def test_version_option_reports_installed_distribution_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "2.0.0"
