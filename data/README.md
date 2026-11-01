# Data directory

## Layout

| Path | Role |
|------|------|
| `raw/bilingual_product_reviews.csv` | **Primary** curated dataset authored for this project (binary labels) |
| `raw/` | Immutable originals — do not overwrite in place |
| `processed/` | Derived cleaned CSVs written by training / prepare scripts |
| `processed/reviews_train_expanded.csv` | Preferred training set when present (local; gitignored) |
| `raw/external/` | Optional local copies of third-party CSVs (gitignored) |

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
file exists. Build it with:

```powershell
# Curated + openly licensed UCI files (if present under raw/external/)
python scripts/build_expanded_dataset.py

# Workspace-only: also merge local Daraz reference exports (NOT redistributable)
python scripts/prepare_dataset.py --from-references
python scripts/build_expanded_dataset.py --include-local-references
```

A sidecar `reviews_train_expanded.manifest.json` records sources, class counts,
language counts, and whether the build is redistributable.

Open sources (when downloaded via `scripts/prepare_external_test_csvs.py`):

- `uci_roman_urdu_binary.csv` → language `roman_urdu`
- `uci_amazon_imdb_yelp_combined.csv` → language `english`

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
