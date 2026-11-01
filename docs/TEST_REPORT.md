# Test Report

**Date:** 2026-10-27  
**Environment:** Windows 10, Python 3.14.3, project `.venv`  
**CPU-only:** Yes (scikit-learn / no GPU libraries)

---

## Commands executed

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
python -m src.train --compare-svc
python -m src.evaluate
python -m src.linguistic_analysis
pytest
ruff check app.py src tests scripts
python -m compileall -q app.py src tests scripts
```

---

## Pytest results

| Result | Count |
|--------|-------|
| Passed | **71** |
| Failed | **0** |
| Skipped | **0** (AppTest skipped only if model missing) |

Reports isolation: session training writes `holdout_split` / `train_meta` only under a temp `reports_dir`.

### Suites covered

- `test_preprocessing.py` — empty/None/Unicode/negation/URLs/consistency
- `test_data_validation.py` — columns, labels, leakage, missing files
- `test_training.py` — train/save/load, unfitted guard
- `test_prediction.py` — single/batch/CSV, consistency
- `test_evaluation.py` — metrics from predictions, CM shape, no NaN
- `test_app.py` — import, no API keys, CSV paths, AppTest smoke
- `test_integration.py` — full load→train→evaluate→predict path

---

## Ruff / syntax

- `ruff check app.py src tests scripts` → **All checks passed**
- `python -m compileall` → **success**

---

## Dataset used

| Field | Value |
|-------|-------|
| Preferred train path | `data/processed/reviews_train_expanded.csv` (when present) |
| Curated fallback | `data/raw/bilingual_product_reviews.csv` (160 rows, MIT) |
| Languages | english, roman_urdu, code_mixed (+ heuristic restore) |
| Build recipe | `scripts/build_expanded_dataset.py` + `data/README.md` |

Optional reference CSVs audited in `docs/REFERENCE_ANALYSIS.md` (licenses unclear).
Pytest uses `tests/fixtures/synthetic_train.csv` only and writes reports to a
temp directory (does **not** overwrite project `reports/`).

---

## Model artifacts

| File | Status |
|------|--------|
| `models/sentiment_pipeline.joblib` | Present (~45 KB) |
| `models/sentiment_pipeline_linearsvc.joblib` | Present (comparison) |
| Vectorizer | Inside sklearn Pipeline (not a separate file) |

---

## Evaluation metrics (actual)

After retraining on the expanded set, read the live numbers from:

- `reports/evaluation_metrics.json` (includes macro F1, PR-AUC, per-language)
- `reports/evaluation_report.md`
- `reports/model_comparison.md` (when trained with `--compare-svc`)

Do **not** treat older curated-only ~0.84 / 32-row figures as the current
workspace artifact.

---

## Linguistic error analysis

Command: `python -m src.linguistic_analysis`  
Report: `reports/linguistic_error_analysis.md`  
Probes cover negation, contrast (`lekin` / `but`), and informal spelling.
Match rate is qualitative evidence — not a forced accuracy target.

---

## Local application verification

- `import app` succeeds without API keys
- Artifact loads from `models/sentiment_pipeline.joblib`
- Streamlit launch command: `streamlit run app.py` (ready; interactive browser session left to the user)

---

## Deployment

| Item | Status |
|------|--------|
| `docs/DEPLOYMENT.md` | Written |
| Streamlit Community Cloud public URL | **Not verified** — no live demo claimed |

---

## Known bugs / limitations

- Small dataset → metrics not generalizable
- Binary labels only
- Informal spelling variants can still flip predictions
- `holdout_split.joblib` is gitignored; re-run `python -m src.train` to regenerate evaluation holdout after clone if needed

---

## Remaining manual steps for you

1. `streamlit run app.py` and click through Single / Batch / Evaluation / Research  
2. Optional: push to GitHub when ready (ask before any push)  
3. Optional: deploy via Streamlit Cloud and paste the verified URL into README  
4. Optional local experiment: `python scripts/prepare_dataset.py --from-references` (do not publish without license clearance)
