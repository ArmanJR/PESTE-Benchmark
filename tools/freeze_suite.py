"""Maintainer-only one-time suite manifest sealer."""

import argparse
import logging
from pathlib import Path

from peste.constants import DEFAULT_SUITE_ID, PROJECT_ROOT
from peste.dataset import write_initial_manifest
from peste.logging import configure_logging
from peste.specs import load_suite

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE_ID)
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "cache" / DEFAULT_SUITE_ID)
    arguments = parser.parse_args()
    configure_logging()
    suite = load_suite(arguments.suite)
    digest = write_initial_manifest(
        suite,
        PROJECT_ROOT / "suites" / suite.suite_id,
        arguments.cache,
    )
    LOGGER.info(
        "Set suite.json manifest_sha256 to the logged digest before any benchmark run",
        extra={"manifest_sha256": digest},
    )


if __name__ == "__main__":
    main()
