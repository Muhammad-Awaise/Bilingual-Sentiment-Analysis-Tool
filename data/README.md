# Data directory

## Layout

| Path | Role |
|------|------|
| `raw/bilingual_product_reviews.csv` | **Primary** curated dataset authored for this project (binary labels) |
| `raw/` | Immutable originals — do not overwrite in place |
| `processed/` | Derived cleaned CSVs written by training / prepare scripts |
| `processed/reviews_train_expanded.csv` | Preferred large training set (tracked so clones can retrain) |
| `raw/external/` | Prepared third-party CSVs (tracked; `downloads/` extracts stay local) |

## Primary dataset

- **File:** `raw/bilingual_product_reviews.csv`
- **Columns:** `text`, `label`, `language`
- **Labels:** `positive`, `negative` only (binary)
- **Languages:** `english`, `roman_urdu`, `code_mixed`
- **Authorship:** Original example reviews written for this repository (product-review style)
- **License:** Same as the project MIT License for this CSV
- **Limitation:** Small size — suitable for demonstrating the pipeline and linguistic probes, **not** for claiming large-scale generalization

## Expanded training set (build recipe)

`python -m src.train` prefers `processed/reviews_train_expanded.csv` when that
file exists. Build the **large** workspace set with:

```powershell
# Curated + UCI/Kaggle/TweetEval (if present) + synthetic Urdu-script corpus
python scripts/build_expanded_dataset.py

# Max local set: also merge Daraz reference exports (NOT redistributable)
python scripts/prepare_dataset.py --from-references
python scripts/build_expanded_dataset.py --include-local-references
```

A sidecar `reviews_train_expanded.manifest.json` records sources, class counts,
language counts, and whether the build is redistributable.

The synthetic Urdu-script corpus is included by default so Nastaliq reviews
cannot collapse to a constant negative prior again.

Open / research sources (when present under `raw/external/`):

- `uci_roman_urdu_binary.csv` → language `roman_urdu`
- `kaggle_roman_urdu_binary_11000.csv` → language `roman_urdu`
- `uci_amazon_imdb_yelp_combined.csv` → language `english`
- `tweeteval_sentiment_binary.csv` → language `english`

Missing language tags are restored with `src.data_loader.infer_language`.

## Optional third-party sources (local only)

Reference repositories under `references/` contain larger review CSVs. Their
**redistribution rights are unclear** (see `docs/REFERENCE_ANALYSIS.md`).

If you have the right to use them locally for experiments:

```powershell
python scripts/prepare_dataset.py --from-references
```

This writes a prepared file under `data/processed/` and **excludes Neutral**
rows so labels stay binary. Do **not** commit those outputs to a public GitHub
repo unless you have verified the dataset license.
