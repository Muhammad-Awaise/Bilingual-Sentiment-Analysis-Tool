"""Experiment metadata, artifact fingerprints, and stale-artifact checks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import MODEL_ARTIFACT_PATH, REPORTS_DIR, TRAIN_META_PATH


def file_fingerprint(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_run_metadata(
    *,
    data_path: str | Path | None,
    n_records: int,
    class_counts: dict[str, int],
    language_counts: dict[str, Any],
    provenance_counts: dict[str, Any] | None = None,
    source_counts: dict[str, Any] | None = None,
    random_seed: int,
    preprocessing: dict[str, Any],
    vectorizer: dict[str, Any],
    classifier: dict[str, Any],
    calibration: dict[str, Any],
    artifact_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_path = Path(data_path) if data_path else None
    artifact = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    meta: dict[str, Any] = {
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path) if data_path else None,
        "dataset_fingerprint": file_fingerprint(data_path) if data_path else None,
        "dataset_version": (
            data_path.with_suffix(".manifest.json").name
            if data_path and data_path.with_suffix(".manifest.json").exists()
            else None
        ),
        "n_records": int(n_records),
        "class_distribution": class_counts,
        "language_distribution": {
            str(k): int(v) for k, v in (language_counts or {}).items()
        },
        "provenance_distribution": {
            str(k): int(v) for k, v in (provenance_counts or {}).items()
        },
        "source_distribution": {
            str(k): int(v) for k, v in (source_counts or {}).items()
        },
        "random_seed": int(random_seed),
        "preprocessing_configuration": preprocessing,
        "vectorizer_parameters": vectorizer,
        "classifier_parameters": classifier,
        "calibration_method": calibration,
        "artifact_path": str(artifact),
        "artifact_fingerprint": file_fingerprint(artifact) if artifact.exists() else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "evaluation_metrics": metrics,
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(meta: dict[str, Any], path: str | Path | None = None) -> Path:
    out = Path(path) if path else REPORTS_DIR / TRAIN_META_PATH.name
    out.parent.mkdir(parents=True, exist_ok=True)
    # Refresh artifact fingerprint after dump
    art = meta.get("artifact_path")
    if art:
        meta = {**meta, "artifact_fingerprint": file_fingerprint(art)}
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def stale_artifact_report(
    artifact_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    train_meta_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Detect mismatched model ↔ metrics ↔ train_meta fingerprints.

    Never silently treats an old model as current after a failed retrain —
    callers should surface warnings to the UI.
    """
    artifact = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    metrics = Path(metrics_path) if metrics_path else REPORTS_DIR / "evaluation_metrics.json"
    train_meta = Path(train_meta_path) if train_meta_path else REPORTS_DIR / TRAIN_META_PATH.name

    warnings: list[str] = []
    art_fp = file_fingerprint(artifact)
    meta_fp = None
    recorded_fp = None
    if train_meta.exists():
        try:
            recorded = json.loads(train_meta.read_text(encoding="utf-8"))
            recorded_fp = recorded.get("artifact_fingerprint")
            meta_fp = recorded_fp
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read train_meta: {exc}")

    if artifact.exists() and recorded_fp and art_fp and recorded_fp != art_fp:
        warnings.append(
            "Model artifact fingerprint does not match train_meta.json — "
            "reports may describe a different training run."
        )
    if artifact.exists() and metrics.exists():
        if artifact.stat().st_mtime > metrics.stat().st_mtime + 5:
            warnings.append(
                "Model artifact is newer than evaluation_metrics.json. "
                "Re-run evaluation; do not trust stale metrics."
            )
        if metrics.stat().st_mtime > artifact.stat().st_mtime + 5 and not recorded_fp:
            warnings.append(
                "Evaluation metrics are newer than the model without matching "
                "train_meta fingerprint — verify the evaluation target."
            )
    if not artifact.exists():
        warnings.append("Model artifact missing.")

    return {
        "artifact_fingerprint": art_fp,
        "train_meta_artifact_fingerprint": meta_fp,
        "warnings": warnings,
        "is_stale": bool(warnings),
    }
