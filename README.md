# Bilingual Sentiment Analyzer

**English and Roman Urdu product reviews** — a CPU-only, offline-friendly Streamlit app backed by TF-IDF + calibrated LinearSVC (LogReg comparison included).

## Research question

> How do negation, code-switching, and informal Roman Urdu spelling affect the performance of sentiment classification models on bilingual product reviews?

This project connects **English Linguistics** (negation, contrast, language variation, contact) with **computational methods** (feature-based ML, evaluation, error analysis).

## Features

- Single-review prediction (English / Roman Urdu / code-mixed)
- Batch CSV upload with downloadable results
- Sentiment distribution charts
- Held-out evaluation metrics and confusion matrix
- Research notes for non-specialist readers
- No paid APIs, no GPU, no LLM required for predictions

## Architecture

```
Review text
  → shared clean_text (negation/contrast preserved; light nahin→nahi map)
  → TF-IDF (1–2 grams, fit on training data only)
  → LinearSVC (class_weight=balanced) + sigmoid calibration
  → joblib Pipeline artifact
  → Streamlit UI (app.py)
```

Core package: `src/` (`data_loader`, `preprocessing`, `train`, `evaluate`, `predict`, `linguistic_analysis`).

## Dataset provenance and license

| Dataset | Path | License / notes |
|---------|------|-----------------|
| **Primary (shipped)** | `data/raw/bilingual_product_reviews.csv` | Original curated reviews for this project; MIT with the code |
| Size | 160 rows | 79 positive, 81 negative |
| Languages | english, roman_urdu, code_mixed | Product-review style |
| Labels | positive / negative / **neutral** | Curated set includes Neutral + Urdu-script rows |

**Important:** Metrics on this small set demonstrate the pipeline. They are **not** large-scale benchmarks.

Reference repositories under `references/` were inspected for ideas only. Their datasets/models are **not redistributed** here (unclear licenses). See `docs/REFERENCE_ANALYSIS.md`.

## Installation (Windows PowerShell)

```powershell
cd "C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Prefers `data/processed/reviews_train_expanded.csv` when present (see
`data/README.md` for the build recipe); otherwise uses the curated 160-row CSV.

```powershell
python -m src.train --compare-svc
# optional hyperparameter search (stratified CV on F1-macro):
python -m src.train --tune --compare-svc
# force Logistic Regression as the primary artifact instead:
python -m src.train --classifier logreg --compare-svc
```

Training uses `class_weight="balanced"`, sigmoid probability calibration, and
shared preprocessing that preserves negation/contrast (with a light closed-set
`nahin→nahi` map). Default primary model is LinearSVC; `--compare-svc` also
fits LogReg and writes `reports/model_comparison.md`.

## Evaluate

```powershell
python -m src.evaluate
python -m src.linguistic_analysis
```

Reports are written to `reports/` (macro F1, PR-AUC, per-language slices, and
optional `model_comparison.md`).

## Run the app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Run tests

```powershell
pytest
ruff check app.py src tests scripts
```

## Actual model results (this workspace)

Computed on a stratified held-out test set after training on the **expanded**
dataset (`data/processed/reviews_train_expanded.csv`), then
`python -m src.evaluate`. **Not hardcoded** — re-run train/evaluate after
changing data or code, then refresh this table from `reports/`.

| Metric | Value |
|--------|-------|
| Model | TF-IDF + LinearSVC (balanced, calibrated) |
| Data | Expanded ~35k rows (`reviews_train_expanded.manifest.json`) |
| Train / test size | 28150 / 7038 |
| Accuracy | 0.890 |
| Precision (macro) | 0.883 |
| Recall (macro) | 0.886 |
| F1 (macro) | 0.885 |
| PR-AUC (positive) | 0.975 |
| Majority-class baseline accuracy | 0.616 |
| Δ F1 vs majority baseline | ~0.269 |
| Per-language F1 (macro) | code_mixed ~0.95 · roman_urdu ~0.82 · english ~0.82 |
| Linguistic probes | **15/15** (`reports/linguistic_error_analysis.md`) |

Full JSON: `reports/evaluation_metrics.json`.  
Model comparison: `reports/model_comparison.md`.

**Curated-only appendix:** training on the MIT curated CSV alone historically
yielded ~0.84 macro F1 on a 32-row holdout (majority baseline 0.50). That
demonstrates the pipeline on redistributable data; it is **not** the current
workspace production artifact.

## Screenshots

After launching the app, capture:

1. Single-review prediction with a code-mixed example  
2. Batch CSV results + bar chart  
3. Evaluation tab with metrics  

## Known limitations

- Expanded metrics can look strong while linguistic probes still fail on hard cases
- Binary labels only (no Neutral)
- Bag-of-ngrams cannot model deep discourse or sarcasm
- Calibrated scores are approximate probabilities, not guarantees
- Roman Urdu spelling is only lightly normalized (closed-set negation map)
- Local Daraz/Kaggle reference rows are optional and not claimed as redistributable

## Reference repositories and acknowledgements

Inspected (architecture inspiration only — code not copied):

1. [ahsaanjameel595/roman_urdu_sentiment_analysis](https://github.com/ahsaanjameel595/roman_urdu_sentiment_analysis)
2. [Gardezi-Enterprises/AIML-Bilingual-Sentiment-Analysis-of-Daraz-Reviews](https://github.com/Gardezi-Enterprises/AIML-Bilingual-Sentiment-Analysis-of-Daraz-Reviews)

Their reported metrics are **theirs**, not this project's results.

## Future improvements

- Stronger discourse / negation-scope features beyond bag-of-ngrams
- Optional Neutral class with a clear annotation guide
- Verified Streamlit Cloud demo URL
- Optional transformer baseline for comparison

## Live demo

Not verified in this workspace. See `docs/DEPLOYMENT.md` to deploy on Streamlit Community Cloud when ready.

## Documentation map

| Doc | Contents |
|-----|----------|
| `docs/REFERENCE_ANALYSIS.md` | Third-party audit |
| `docs/PROJECT_PLAN.md` | Scope, milestones, interview story |
| `docs/LINGUISTIC_DECISIONS.md` | Why preprocessing keeps negation |
| `docs/TEST_REPORT.md` | Test run evidence |
| `docs/DEPLOYMENT.md` | Cloud steps |

## License

MIT for **this project's original code and curated CSV**. Third-party reference materials remain under their own (often unclear) terms — do not assume permission to republish them.
