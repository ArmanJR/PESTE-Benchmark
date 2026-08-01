"""Benchmark-wide immutable constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PESTE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_SUITE_ID = "fleurs-fa-ir-v1"
DEFAULT_NORMALIZER = "fa-v1"
DATASET_REPOSITORY = "google/fleurs"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATASET_CONFIG = "fa_ir"
EXPECTED_SPLIT_COUNTS = {"train": 3101, "validation": 369, "test": 871}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
RANKED_SPLIT = "test"
DEFAULT_SEED = 20250731
