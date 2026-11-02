# Evaluation Report

- **Model:** TF-IDF + LinearSVC (calibrated)
- **Data path:** `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\data\processed\reviews_train_expanded.csv`
- **Train size:** 70925
- **Test size:** 15116
- **Class counts (full data):** {'positive': 47106, 'negative': 28180, 'neutral': 293}
- **Language counts (full data):** {'english': 28596, 'roman_urdu': 25899, 'code_mixed': 15395, 'urdu': 5070, 'unknown': 619}
- **Class weight:** balanced
- **Calibrated:** True
- **Tuned:** False

## Metrics (held-out test set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8782 |
| Precision (macro) | 0.9081 |
| Recall (macro) | 0.8946 |
| F1 (macro) | 0.9012 |
| F1 (weighted) | 0.8779 |
| Majority-class baseline accuracy | 0.6232 |
| Δ F1 vs majority baseline | 0.2779 |
| PR-AUC (positive) | 0.9676 |

Confusion matrix saved to `C:\Users\GrowMore Devs\Desktop\Projects\Bilingual-Sentiment-Analysis-Tool\reports\confusion_matrix.png`.

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| negative | 0.8450 | 0.8251 | 0.8349 | 5636 |
| neutral | 0.9825 | 0.9492 | 0.9655 | 59 |
| positive | 0.8967 | 0.9096 | 0.9031 | 9421 |

## Neutral analysis

- Support: 59
- Precision / Recall / F1: 0.9825 / 0.9492 / 0.9655
- Gold Neutral → other: {'negative': 1, 'positive': 2}
- Other → predicted Neutral: {'negative': 1}

## Per-language metrics

| Language | N | Accuracy | F1 (macro) |
|----------|---|----------|------------|
| code_mixed | 3070 | 0.9625 | 0.8606 |
| english | 5700 | 0.8604 | 0.5539 |
| roman_urdu | 5205 | 0.8223 | 0.5472 |
| unknown | 116 | 0.9569 | 0.6373 |
| urdu | 1025 | 1.0000 | 1.0000 |

## Calibration

- Brier score: 0.1752
- ECE: 0.0080
- Mean confidence: 0.8714638377329366
- Note: Scores are sigmoid-calibrated class probabilities when the artifact used CalibratedClassifierCV. They are approximate confidence, not certainty about human sentiment.

## Challenge set

- N: 29 | correct: 27 | incorrect: 2 | accuracy: 0.9310
- Note: Challenge-set labels are linguistic probes for error analysis. The model is not forced to predict the expected label.

## Limitations

Metrics are computed on a held-out stratified test set created during training. They are not hardcoded. Macro F1 and PR-AUC are preferred over accuracy when classes are imbalanced. Small curated subsets and synthetic Urdu templates can yield optimistic language-slice scores — check per_provenance. Calibration scores are approximate confidence, not certainty.
