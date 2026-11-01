# Project Plan — Bilingual Sentiment Analyzer

**Audience:** Graduate-admissions-ready linguistics + NLP project  
**Constraint:** Four days, CPU-only, no paid APIs  
**Research question:** How do negation, code-switching, and informal Roman Urdu spelling affect sentiment classification on bilingual product reviews?

---

## 1. Smallest feasible deliverable (chosen)

A **binary** (positive / negative) TF-IDF + Logistic Regression pipeline with:

1. Shared preprocessing that **keeps** negation and contrast markers  
2. Reproducible train / evaluate / save / load  
3. Streamlit single-review + CSV batch UI  
4. Linguistic error-analysis fixtures (behavior tests, not forced accuracy)  
5. pytest coverage and docs for licensing / limitations  

**Deferred (if time remains):** LinearSVC comparison UI toggle, language-ID breakdown charts, three-class Neutral support.

---

## 2. Why binary classification?

- Reference 1 is binary (`pos`/`neg`).
- Daraz files include Neutral, but combining Neutral with binary-only sources without a shared annotation guide would be scientifically sloppy.
- Four-day scope favors one clear label set we can defend in interviews.
- Neutral rows are **excluded** (and documented) if optional Daraz preparation is used.

---

## 3. Dataset strategy

| Layer | Location | Purpose |
|-------|----------|---------|
| Original curated set | `data/raw/bilingual_product_reviews.csv` | Shippable, redistributable training data authored for this project |
| Optional external prep | `scripts/prepare_dataset.py` | Local-only merge from `references/` CSVs → `data/processed/` (gitignored large outputs) |
| Test fixtures | `tests/fixtures/` | Tiny synthetic sets for CI; **not** research claims |
| Immutable raw | `data/raw/` | Do not overwrite; derived files go to `data/processed/` |

The curated set covers English, Roman Urdu, and code-mixed product reviews with explicit labels. It is **small** — metrics demonstrate the pipeline, not large-scale generalization.

---

## 4. Architecture

```
CSV / text input
    → preprocessing.clean_text (same at train & infer)
    → TfidfVectorizer (1–2 grams, fit on train only)
    → LogisticRegression (or majority baseline for comparison)
    → joblib pipeline artifact in models/
    → Streamlit (app.py) / evaluate.py / linguistic_analysis.py
```

Prevention of leakage: stratified split **before** fitting TF-IDF; optional near-duplicate text drop; test set used only for final metrics.

---

## 5. Four-day milestones

| Day | Focus | Acceptance |
|-----|-------|------------|
| 1 | Structure, docs, data load, preprocess, train, unit tests | Train on CPU; first tests pass |
| 2 | Evaluate, save model, linguistic fixtures, error report | Reload + predict; real metrics file |
| 3 | Streamlit single + batch + charts + validation | Local app classifies reviews |
| 4 | Full pytest, README, DEPLOYMENT, TEST_REPORT | No failing mandatory tests |

---

## 6. Interview-friendly technical story

1. **Linguistics:** Negation and contrast change polarity; Roman Urdu spelling varies; code-switching mixes lexicons.  
2. **Method:** Lightweight bag-of-ngrams features + linear classifier — transparent and CPU-friendly.  
3. **Honesty:** Small data → limited generalization; model scores ≠ calibrated probabilities; predictions ≠ ground-truth meaning.

---

## 7. Risk register

| Risk | Mitigation |
|------|------------|
| No redistributable large dataset | Ship curated CSV; optional local prep |
| Reference stopwords delete negation | Custom conservative preprocessing |
| Overclaiming accuracy | Never hardcode metrics; write from evaluation |
| Streamlit Cloud missing artifacts | Document train step; keep model small enough to commit if license OK |
