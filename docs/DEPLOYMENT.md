# Deployment Guide — Streamlit Community Cloud

Local execution is **mandatory** and already works without the cloud. Public
deployment is optional.

## Prerequisites

- A GitHub account and a public (or connected private) repository containing this project
- Streamlit Community Cloud account: https://share.streamlit.io/
- Committed files: `app.py`, `requirements.txt`, `src/`, curated `data/raw/`, and
  `models/sentiment_pipeline.joblib` (or a documented train step after clone)

## Steps

1. Push this project to GitHub (only when you choose to — do not publish private data).
2. Open Streamlit Community Cloud → **New app**.
3. Select the repository, branch, and main file: `app.py`.
4. Confirm Python version compatible with `requirements.txt` (3.11–3.14 recommended).
5. Deploy and wait for the build log to finish.

## Paths

All paths resolve from the repository root via `src/config.py`. Do not hardcode
absolute Windows paths in cloud config.

## Artifacts

| Artifact | Cloud note |
|----------|------------|
| `models/sentiment_pipeline.joblib` | Prefer committing the small trained pipeline so the app predicts immediately |
| `reports/evaluation_metrics.json` | Optional; Evaluation tab shows a message if missing |
| Large Daraz / Kaggle CSVs | **Do not** upload unless redistribution is allowed |

If the model file cannot be committed, document that users (or a cloud build
script you add later) must run `python -m src.train` — Streamlit Cloud does not
run training by default.

## What not to do

- No paid APIs, secrets, or GPU services
- No scraping at runtime
- Do not claim a live demo URL until you open it and verify predictions work

## Verification checklist

- [ ] App loads without API keys
- [ ] Single-review prediction returns positive/negative
- [ ] Batch CSV upload works on a tiny file
- [ ] Research tab renders
- [ ] Public URL recorded in README only after manual verification

**Deployment status for this workspace:** Not verified (no public URL claimed).
