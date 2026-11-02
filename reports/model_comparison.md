# Model Comparison (same holdout)

- **Holdout size:** 15233
- **Majority baseline accuracy:** 0.6248
- **Preferred (by macro F1):** linearsvc

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | PR-AUC (pos) |
|-------|----------|-------------------|----------------|------------|--------------|
| TF-IDF + Logistic Regression (proba) | 0.8783 | 0.8845 | 0.9081 | 0.8953 | 0.9708 |
| TF-IDF + LinearSVC (calibrated) | 0.8809 | 0.9091 | 0.8919 | 0.9002 | 0.9694 |

Both models share the same train/test split and preprocessing.
