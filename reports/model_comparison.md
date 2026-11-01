# Model Comparison (same holdout)

- **Holdout size:** 7038
- **Majority baseline accuracy:** 0.6158
- **Preferred (by macro F1):** linearsvc

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | PR-AUC (pos) |
|-------|----------|-------------------|----------------|------------|--------------|
| TF-IDF + Logistic Regression (calibrated) | 0.8865 | 0.8800 | 0.8801 | 0.8801 | 0.9740 |
| TF-IDF + LinearSVC (calibrated) | 0.8905 | 0.8835 | 0.8858 | 0.8846 | 0.9747 |

Both models share the same train/test split and preprocessing.
