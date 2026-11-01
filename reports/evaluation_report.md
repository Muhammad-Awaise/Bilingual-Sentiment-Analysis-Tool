# Evaluation Report

- **Model:** TF-IDF + LinearSVC (calibrated)
- **Data path:** `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\data\processed\reviews_train_expanded.csv`
- **Train size:** 28150
- **Test size:** 7038
- **Class counts (full data):** {'positive': 21667, 'negative': 13521}
- **Language counts (full data):** {'code_mixed': 15884, 'roman_urdu': 14988, 'english': 3696, 'unknown': 620}
- **Class weight:** balanced
- **Calibrated:** True
- **Tuned:** False

## Metrics (held-out test set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8905 |
| Precision (macro) | 0.8835 |
| Recall (macro) | 0.8858 |
| F1 (macro) | 0.8846 |
| Majority-class baseline accuracy | 0.6158 |
| Δ F1 vs majority baseline | 0.2688 |
| PR-AUC (positive) | 0.9747 |

Confusion matrix saved to `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\reports\confusion_matrix.png`.

## Per-language metrics

| Language | N | Accuracy | F1 (macro) |
|----------|---|----------|------------|
| code_mixed | 3188 | 0.9561 | 0.9491 |
| english | 764 | 0.8325 | 0.8324 |
| roman_urdu | 2955 | 0.8321 | 0.8295 |
| unknown | 131 | 0.9466 | 0.9450 |

## Limitations

Metrics are computed on a held-out stratified test set created during training. They are not hardcoded. Macro F1 and PR-AUC are preferred over accuracy when classes are imbalanced. Small curated subsets yield optimistic or unstable estimates and should not be treated as large-scale benchmarks.
