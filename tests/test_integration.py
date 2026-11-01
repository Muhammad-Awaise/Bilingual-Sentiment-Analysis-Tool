"""End-to-end pipeline on the synthetic fixture (offline)."""

from pathlib import Path

import joblib

from src.config import SUPPORTED_LABELS
from src.data_loader import class_counts, load_reviews_csv
from src.evaluate import compute_metrics
from src.predict import predict_one
from src.train import train_and_save


def test_full_pipeline(synthetic_csv: Path, tmp_path: Path):
    df = load_reviews_csv(synthetic_csv)
    counts = class_counts(df)
    assert counts["positive"] > 0 and counts["negative"] > 0

    artifact = tmp_path / "pipeline.joblib"
    reports = tmp_path / "reports"
    meta = train_and_save(
        data_path=synthetic_csv,
        artifact_path=artifact,
        reports_dir=reports,
        write_processed=False,
    )
    assert artifact.exists()

    payload = joblib.load(artifact)
    pipe = payload["pipeline"]

    # Evaluate on the persisted holdout from training
    split = joblib.load(Path(meta["split_path"]))
    y_pred = pipe.predict(split["X_test"])
    metrics = compute_metrics(split["y_test"], y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["f1_macro"] == metrics["f1_macro"]  # not NaN
    assert "lang_test" in split

    reloaded = predict_one("bohat acha product hai", artifact_path=artifact)
    assert reloaded["prediction"] in SUPPORTED_LABELS
    assert "calibrated" in reloaded
