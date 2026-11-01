# Model Comparison (same holdout)

- **Holdout size:** 7052
- **Majority baseline accuracy:** 0.6151
- **Preferred (by macro F1):** linearsvc

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | PR-AUC (pos) |
|-------|----------|-------------------|----------------|------------|--------------|
| TF-IDF + Logistic Regression (calibrated) | 0.8866 | 0.7156 | 0.8646 | 0.7626 | 0.9729 |
| TF-IDF + LinearSVC (calibrated) | 0.8897 | 0.8388 | 0.7563 | 0.7892 | 0.9738 |

Both models share the same train/test split and preprocessing.
