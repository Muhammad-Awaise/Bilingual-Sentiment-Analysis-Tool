# External Daraz Evaluation

- **Usable:** True
- **N:** 7998
- **Model:** TF-IDF + LinearSVC (calibrated)

## Metrics

| Accuracy | 0.5458 |
| Macro precision | 0.6652 |
| Macro recall | 0.6403 |
| Macro F1 | 0.4541 |

## Δ vs internal holdout

{'accuracy': -0.3324470804126012, 'f1_macro': -0.4470666826537814, 'precision_macro': -0.24286697380744204, 'recall_macro': -0.2542653708930408}

## Limitations

- Daraz labels are reference_labelled unless manually annotated.
- External Daraz metrics must not be compared as if they were in-distribution holdout scores.
