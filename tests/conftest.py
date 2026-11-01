"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.train import train_and_save

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_CSV = FIXTURES / "synthetic_train.csv"


@pytest.fixture(scope="session")
def synthetic_csv() -> Path:
    return SYNTHETIC_CSV


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory, synthetic_csv: Path) -> Path:
    """Train once per test session on the tiny synthetic fixture.

    Reports (holdout split / train_meta) stay inside the temp tree so pytest
    never overwrites project reports/.
    """
    out_dir = tmp_path_factory.mktemp("model")
    artifact = out_dir / "pipeline.joblib"
    reports_dir = out_dir / "reports"
    train_and_save(
        data_path=synthetic_csv,
        artifact_path=artifact,
        classifier="logreg",
        reports_dir=reports_dir,
        calibrate=True,
        tune=False,
        write_processed=False,
    )
    return artifact


@pytest.fixture
def project_root() -> Path:
    return ROOT
