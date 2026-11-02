# External datasets for local testing

These files were **downloaded from public sources** (not invented).
Use locally for testing. Check each license before publishing/redistributing.

| File | Rows | Source | Labels |
|------|------|--------|--------|
| `uci_roman_urdu_sentiment.csv` | 19626 | UCI ML Repository id=458 (Sharf, 2017) | `{'positive': 5854, 'neutral': 8533, 'negative': 5239}` |
| `uci_roman_urdu_binary.csv` | 11093 | UCI Roman Urdu binary subset (Neutral dropped) | `{'positive': 5854, 'negative': 5239}` |
| `kaggle_roman_urdu_binary_11000.csv` | 10983 | From reference zip (Kaggle/RUSAD-style binary Roman Urdu) | `{'negative': 5298, 'positive': 5685}` |
| `uci_amazon_cells_labelled.csv` | 990 | UCI Sentiment Labelled Sentences (amazon_cells_labelled.txt) | `{'negative': 497, 'positive': 493}` |
| `uci_imdb_labelled.csv` | 997 | UCI Sentiment Labelled Sentences (imdb_labelled.txt) | `{'negative': 499, 'positive': 498}` |
| `uci_yelp_labelled.csv` | 996 | UCI Sentiment Labelled Sentences (yelp_labelled.txt) | `{'positive': 499, 'negative': 497}` |
| `uci_amazon_imdb_yelp_combined.csv` | 2982 | UCI Amazon + IMDB + Yelp combined | `{'negative': 1492, 'positive': 1490}` |
| `tweeteval_sentiment.csv` | 45586 | TweetEval sentiment train split (CardiffNLP) | `{'positive': 17840, 'neutral': 20655, 'negative': 7091}` |
| `tweeteval_sentiment_binary.csv` | 24931 | TweetEval binary subset | `{'positive': 17840, 'negative': 7091}` |

## Already present (Daraz reference)

- `Daraz_FYP_Dataset.csv`
- `daraz_code_mixed_product_reviews.csv`
- `../daraz_batch_test_4000.csv`

## How to use in the app

1. Open Batch CSV tab and upload one file
2. Select text column: `review`
3. Gold labels are in `gold_label` (used by Evaluation batch section)

## Notes

- The app model predicts only positive/negative.
- Prefer `*_binary.csv` files for fair scoring against this model.
