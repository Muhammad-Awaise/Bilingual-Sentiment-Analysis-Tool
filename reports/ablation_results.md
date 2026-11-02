# Ablation Results

- Created: 2026-11-02T06:23:45.961293+00:00
- Random seed: 42
- Rows used: 8000 / 75579
- Subsampled: True

| Configuration | Accuracy | Macro F1 | Neutral F1 | EN F1 | RU F1 | CM F1 | Urdu F1 | Train s |
|---------------|----------|----------|------------|-------|-------|-------|---------|---------|
| word_tfidf_only | 0.8488 | 0.8611 | 0.9091 | 0.5103 | 0.5350 | 0.6291 | 1.0000 | 6.258 |
| char_tfidf_only | 0.8213 | 0.8398 | 0.9091 | 0.4849 | 0.5157 | 0.6014 | 1.0000 | 7.145 |
| word_char_tfidf | 0.8375 | 0.8526 | 0.9091 | 0.5001 | 0.5225 | 0.6272 | 1.0000 | 6.823 |
| word_char_negation | 0.8438 | 0.8572 | 0.9091 | 0.5087 | 0.5274 | 0.6270 | 1.0000 | 4.168 |
| word_char_contrast | 0.8375 | 0.8523 | 0.9091 | 0.4941 | 0.5262 | 0.6291 | 1.0000 | 4.343 |
| word_char_negation_contrast | 0.8375 | 0.8526 | 0.9091 | 0.5001 | 0.5225 | 0.6272 | 1.0000 | 4.971 |
| baseline_preproc | 0.8381 | 0.8529 | 0.9091 | 0.4925 | 0.5299 | 0.6267 | 1.0000 | 3.278 |
| nahi_normalization | 0.8363 | 0.8515 | 0.9091 | 0.4925 | 0.5274 | 0.6244 | 1.0000 | 3.511 |
| contrast_marker_only | 0.8438 | 0.8572 | 0.9091 | 0.5076 | 0.5297 | 0.6246 | 1.0000 | 4.448 |
| full_linguistic | 0.8375 | 0.8526 | 0.9091 | 0.5001 | 0.5225 | 0.6272 | 1.0000 | 4.576 |

All metrics computed on the same holdout. Values are not hardcoded. Subsampled runs are for relative comparison only.
