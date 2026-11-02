# Evaluation Report

- **Model:** TF-IDF + LinearSVC (calibrated)
- **Data path:** `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\data\processed\reviews_train_expanded.csv`
- **Train size:** 72525
- **Test size:** 15233
- **Class counts (full data):** {'positive': 47587, 'negative': 28284, 'neutral': 293}
- **Language counts (full data):** {'english': 28637, 'roman_urdu': 25958, 'code_mixed': 15879, 'urdu': 5070, 'unknown': 620}
- **Class weight:** balanced
- **Calibrated:** True
- **Tuned:** False

## Metrics (held-out test set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8809 |
| Precision (macro) | 0.9091 |
| Recall (macro) | 0.8919 |
| F1 (macro) | 0.9002 |
| Majority-class baseline accuracy | 0.6248 |
| Δ F1 vs majority baseline | 0.2755 |
| PR-AUC (positive) | 0.9694 |

Confusion matrix saved to `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\reports\confusion_matrix.png`.

## Per-language metrics

| Language | N | Accuracy | F1 (macro) |
|----------|---|----------|------------|
| code_mixed | 3152 | 0.9613 | 0.6373 |
| english | 5740 | 0.8585 | 0.8865 |
| roman_urdu | 5160 | 0.8302 | 0.5529 |
| unknown | 138 | 0.9638 | 0.6412 |
| urdu | 1043 | 1.0000 | 1.0000 |

## Limitations

Metrics are computed on a held-out stratified test set created during training. They are not hardcoded. Macro F1 and PR-AUC are preferred over accuracy when classes are imbalanced. Small curated subsets yield optimistic or unstable estimates and should not be treated as large-scale benchmarks.
