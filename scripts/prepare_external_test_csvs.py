"""Prepare Streamlit-ready CSVs from downloaded public datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "external"
DL = OUT / "downloads"


def norm_label(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in {"pos", "positive", "1"}:
        return "positive"
    if s in {"neg", "negative", "0"}:
        return "negative"
    if s in {"neu", "neutral", "2"}:
        return "neutral"
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[str, int, str, dict]] = []

    # 1) UCI Roman Urdu
    uci = pd.read_csv(
        DL / "uci_roman_urdu" / "Roman Urdu DataSet.csv",
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    uci = pd.DataFrame({"review": uci[0].astype(str), "gold_label_raw": uci[1]})
    uci["gold_label"] = uci["gold_label_raw"].map(norm_label)
    uci = uci.dropna(subset=["gold_label"])
    uci = uci[uci["review"].str.strip() != ""]
    uci = uci.drop_duplicates(subset=["review"]).reset_index(drop=True)
    uci.insert(0, "review_id", range(1, len(uci) + 1))
    uci[["review_id", "review", "gold_label"]].to_csv(
        OUT / "uci_roman_urdu_sentiment.csv", index=False, encoding="utf-8-sig"
    )
    uci_bin = uci[uci["gold_label"].isin(["positive", "negative"])].copy()
    uci_bin["review_id"] = range(1, len(uci_bin) + 1)
    uci_bin[["review_id", "review", "gold_label"]].to_csv(
        OUT / "uci_roman_urdu_binary.csv", index=False, encoding="utf-8-sig"
    )
    manifest.append(
        (
            "uci_roman_urdu_sentiment.csv",
            len(uci),
            "UCI ML Repository id=458 (Sharf, 2017)",
            dict(Counter(uci["gold_label"])),
        )
    )
    manifest.append(
        (
            "uci_roman_urdu_binary.csv",
            len(uci_bin),
            "UCI Roman Urdu binary subset (Neutral dropped)",
            dict(Counter(uci_bin["gold_label"])),
        )
    )

    # 2) Reference zip / Kaggle-style binary
    ref = next((DL / "ref1_roman_urdu_zip").rglob("RomanUrduSentiment.csv"))
    r = pd.read_csv(ref, header=None, encoding="utf-8", on_bad_lines="skip")
    rdf = pd.DataFrame(
        {"review": r[1].astype(str), "gold_label": r[0].map(norm_label)}
    )
    rdf = rdf.dropna(subset=["gold_label"])
    rdf = rdf[rdf["review"].str.strip() != ""]
    rdf = rdf.drop_duplicates(subset=["review"]).reset_index(drop=True)
    rdf.insert(0, "review_id", range(1, len(rdf) + 1))
    rdf.to_csv(
        OUT / "kaggle_roman_urdu_binary_11000.csv", index=False, encoding="utf-8-sig"
    )
    manifest.append(
        (
            "kaggle_roman_urdu_binary_11000.csv",
            len(rdf),
            "From reference zip (Kaggle/RUSAD-style binary Roman Urdu)",
            dict(Counter(rdf["gold_label"])),
        )
    )

    # 3) UCI Amazon / IMDB / Yelp
    base = DL / "uci_sentiment_labelled"
    mapping = {
        "amazon_cells_labelled.txt": "uci_amazon_cells_labelled.csv",
        "imdb_labelled.txt": "uci_imdb_labelled.csv",
        "yelp_labelled.txt": "uci_yelp_labelled.csv",
    }
    eng_parts = []
    for src_name, out_name in mapping.items():
        src = next(base.rglob(src_name))
        rows = []
        with src.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                text, lab = line.rsplit("\t", 1)
                g = norm_label(lab)
                if g and text.strip():
                    rows.append({"review": text.strip(), "gold_label": g})
        df = pd.DataFrame(rows).drop_duplicates(subset=["review"]).reset_index(drop=True)
        df.insert(0, "review_id", range(1, len(df) + 1))
        df.to_csv(OUT / out_name, index=False, encoding="utf-8-sig")
        eng_parts.append(df)
        manifest.append(
            (
                out_name,
                len(df),
                f"UCI Sentiment Labelled Sentences ({src_name})",
                dict(Counter(df["gold_label"])),
            )
        )

    eng_df = (
        pd.concat(eng_parts, ignore_index=True)
        .drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )
    eng_df["review_id"] = range(1, len(eng_df) + 1)
    eng_df.to_csv(
        OUT / "uci_amazon_imdb_yelp_combined.csv", index=False, encoding="utf-8-sig"
    )
    manifest.append(
        (
            "uci_amazon_imdb_yelp_combined.csv",
            len(eng_df),
            "UCI Amazon + IMDB + Yelp combined",
            dict(Counter(eng_df["gold_label"])),
        )
    )

    # 4) TweetEval
    texts = (
        DL / "tweeteval_sentiment_text.txt"
    ).read_text(encoding="utf-8", errors="replace").splitlines()
    labs = (
        DL / "tweeteval_sentiment_labels.txt"
    ).read_text(encoding="utf-8", errors="replace").splitlines()
    maplab = {"0": "negative", "1": "neutral", "2": "positive"}
    te_rows = []
    for t, lab in zip(texts, labs):
        g = maplab.get(lab.strip())
        if g and t.strip():
            te_rows.append({"review": t.strip(), "gold_label": g})
    te = pd.DataFrame(te_rows).drop_duplicates(subset=["review"]).reset_index(drop=True)
    te.insert(0, "review_id", range(1, len(te) + 1))
    te.to_csv(OUT / "tweeteval_sentiment.csv", index=False, encoding="utf-8-sig")
    te_bin = te[te["gold_label"].isin(["positive", "negative"])].copy()
    te_bin["review_id"] = range(1, len(te_bin) + 1)
    te_bin.to_csv(
        OUT / "tweeteval_sentiment_binary.csv", index=False, encoding="utf-8-sig"
    )
    manifest.append(
        (
            "tweeteval_sentiment.csv",
            len(te),
            "TweetEval sentiment train split (CardiffNLP)",
            dict(Counter(te["gold_label"])),
        )
    )
    manifest.append(
        (
            "tweeteval_sentiment_binary.csv",
            len(te_bin),
            "TweetEval binary subset",
            dict(Counter(te_bin["gold_label"])),
        )
    )

    lines = [
        "# External datasets for local testing",
        "",
        "These files were **downloaded from public sources** (not invented).",
        "Use locally for testing. Check each license before publishing/redistributing.",
        "",
        "| File | Rows | Source | Labels |",
        "|------|------|--------|--------|",
    ]
    for name, n, src, counts in manifest:
        lines.append(f"| `{name}` | {n} | {src} | `{counts}` |")
    lines += [
        "",
        "## Already present (Daraz reference)",
        "",
        "- `Daraz_FYP_Dataset.csv`",
        "- `daraz_code_mixed_product_reviews.csv`",
        "- `../daraz_batch_test_4000.csv`",
        "",
        "## How to use in the app",
        "",
        "1. Open Batch CSV tab and upload one file",
        "2. Select text column: `review`",
        "3. Gold labels are in `gold_label` (used by Evaluation batch section)",
        "",
        "## Notes",
        "",
        "- The app model predicts only positive/negative.",
        "- Prefer `*_binary.csv` files for fair scoring against this model.",
        "",
    ]
    (OUT / "README_EXTERNAL_DATASETS.md").write_text("\n".join(lines), encoding="utf-8")

    print("Prepared files:")
    for name, n, src, counts in manifest:
        print(f"  {name}: {n} rows {counts}")


if __name__ == "__main__":
    main()
