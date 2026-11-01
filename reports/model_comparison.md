# Model Comparison (same holdout)

- **Holdout size:** 7045
- **Majority baseline accuracy:** 0.6153
- **Preferred (by macro F1):** logistic_regression

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | PR-AUC (pos) |
|-------|----------|-------------------|----------------|------------|--------------|
| TF-IDF + Logistic Regression (calibrated) | 0.8883 | 0.6303 | 0.6715 | 0.6439 | 0.9736 |
| TF-IDF + LinearSVC (calibrated) | 0.8884 | 0.5880 | 0.5886 | 0.5883 | 0.9739 |

Both models share the same train/test split and preprocessing.
