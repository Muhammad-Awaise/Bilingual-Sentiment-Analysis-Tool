"""
Bilingual Sentiment Analyzer — Streamlit entry point.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    LABEL_ORDER,
    MAX_BATCH_ROWS,
    MAX_UPLOAD_BYTES,
    METRICS_JSON_PATH,
    MODEL_ARTIFACT_PATH,
    ROOT_DIR,
)
from src.data_loader import normalize_label
from src.evaluate import compute_metrics, confusion_matrix_png_bytes
from src.export_results import (
    build_results_zip,
    dataframe_to_csv_bytes,
    format_single_result_csv,
    make_distribution_figure,
)
from src.ingest import (
    IngestError,
    LABEL_COLUMN_CANDIDATES,
    detect_column_map,
    prepare_for_prediction,
    read_tabular_bytes,
)
from src.predict import (
    PredictionError,
    load_model,
    predict_dataframe,
    predict_one,
    sentiment_summary,
)

st.set_page_config(
    page_title="Bilingual Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
)


_NAV_PAGES = (
    "Single review",
    "Batch upload",
    "Evaluation",
    "Research",
)


def _load_metrics():
    if METRICS_JSON_PATH.exists():
        import json

        return json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
    return None


def _find_gold_column(df: pd.DataFrame) -> str | None:
    """Prefer ingest label aliases; keep working if prediction cols exist too."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    skip = {
        "predicted_sentiment",
        "prediction",
        "model_score_not_probability",
        "model_score",
        "cleaned_review_text",
        "cleaned_text",
        "model_name",
    }
    for name in LABEL_COLUMN_CANDIDATES:
        if name in lower_map and name not in skip:
            return lower_map[name]
    return None


def _confusion_counts_table(cm, labels) -> pd.DataFrame:
    """Compact counts table with short headers (avoids clipping in narrow columns)."""
    matrix = list(cm)
    pred_cols = [f"Pred {str(lab).title()[:3]}." for lab in labels]
    rows = []
    for i, lab in enumerate(labels):
        row = {"True": str(lab).title()}
        for j, col in enumerate(pred_cols):
            row[col] = int(matrix[i][j]) if i < len(matrix) and j < len(matrix[i]) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_from_batch_result(result: pd.DataFrame) -> dict | None:
    """If batch output has gold labels, compute evaluation vs predictions."""
    if result is None or result.empty or "prediction" not in result.columns:
        return None
    gold_col = _find_gold_column(result)
    if gold_col is None:
        return None

    working = result.copy()
    working["_gold"] = working[gold_col].map(normalize_label)
    working = working[working["_gold"].isin(LABEL_ORDER)]
    working = working[working["prediction"].isin(LABEL_ORDER)]
    if working.empty:
        return None

    metrics = compute_metrics(
        working["_gold"].tolist(),
        working["prediction"].tolist(),
        labels=list(LABEL_ORDER),
    )
    metrics["n_compared"] = len(working)
    metrics["gold_column"] = gold_col
    return metrics


def _render_metrics_block(metrics: dict, title: str, source_note: str) -> None:
    st.subheader(title)
    st.caption(source_note)

    baseline = float(metrics.get("majority_baseline_accuracy", 0) or 0)
    accuracy = float(metrics.get("accuracy", 0) or 0)
    labels = metrics.get("labels") or list(LABEL_ORDER)
    cm = metrics.get("confusion_matrix")

    m1, m2, m3, m4 = st.columns(4)
    if baseline:
        m1.metric("Accuracy", f"{accuracy:.3f}", delta=f"{accuracy - baseline:+.3f} vs baseline")
    else:
        m1.metric("Accuracy", f"{accuracy:.3f}")
    m2.metric("Macro F1", f"{float(metrics['f1_macro']):.3f}")
    m3.metric("Precision (macro)", f"{float(metrics['precision_macro']):.3f}")
    m4.metric("Recall (macro)", f"{float(metrics['recall_macro']):.3f}")

    if cm:
        # Tight side-by-side: small gap + near-equal widths so the table is not starved
        # when the sidebar expands (large gaps left a big empty strip beside the chart).
        chart_col, table_col = st.columns([1.15, 1], gap="small")
        with chart_col:
            st.image(
                confusion_matrix_png_bytes(cm, labels),
                caption="Each cell: count + % of that true class (row)",
                use_container_width=True,
            )
        with table_col:
            st.markdown("**Counts table**")
            st.dataframe(
                _confusion_counts_table(cm, labels),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Rows = true · Columns = predicted")

    report = metrics.get("classification_report") or {}
    rows = []
    for label in labels:
        block = report.get(label) or report.get(str(label).lower()) or {}
        if not block:
            continue
        rows.append(
            {
                "Class": str(label).title(),
                "Precision": round(float(block.get("precision", 0)), 3),
                "Recall": round(float(block.get("recall", 0)), 3),
                "F1": round(float(block.get("f1-score", 0)), 3),
                "Support": int(block.get("support", 0)),
            }
        )
    if rows:
        st.markdown("**Per-class scores**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.cache_resource
def _cached_model(path_str: str, mtime_ns: int, size: int):
    """Cache key includes mtime/size so replacing the joblib reloads the model."""
    return load_model(path_str)


def sidebar_status() -> dict | None:
    st.sidebar.title("Bilingual Sentiment Analyzer")
    st.sidebar.caption("English · Roman Urdu · Code-mixed · Urdu-script reviews")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigate")
    st.session_state.setdefault("nav_page", _NAV_PAGES[0])
    st.sidebar.radio(
        "Page",
        options=list(_NAV_PAGES),
        key="nav_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    artifact = MODEL_ARTIFACT_PATH
    if not artifact.exists():
        st.sidebar.error(
            "Model artifact missing. Train locally with:\n\n"
            "`python -m src.train`\n\nthen `python -m src.evaluate`"
        )
        return None
    try:
        stat = artifact.stat()
        payload = _cached_model(str(artifact), stat.st_mtime_ns, stat.st_size)
    except PredictionError as exc:
        st.sidebar.error(str(exc))
        return None

    pipe = payload.get("pipeline")
    has_features = bool(pipe is not None and "features" in getattr(pipe, "named_steps", {}))
    train_size = int(payload.get("train_size") or 0)
    lang_counts = payload.get("language_counts_full") or {}
    urdu_n = int(lang_counts.get("urdu") or 0)

    # Old artifacts collapse every Nastaliq row to ~0.53 negative.
    if train_size < 50000 or urdu_n < 1000 or not has_features:
        st.sidebar.error(
            "Stale model artifact detected. "
            "Urdu-script predictions will be wrong. "
            "Pull latest `models/sentiment_pipeline.joblib`, then click Reload."
        )
        if st.sidebar.button("Reload model cache", type="primary"):
            _cached_model.clear()
            st.rerun()
        return None

    st.sidebar.success("Model loaded (CPU, offline)")
    st.sidebar.write(f"**Model:** {payload.get('model_name', 'unknown')}")
    st.sidebar.write(f"**Train size:** {train_size}")
    st.sidebar.write(f"**Urdu-script rows in train:** {urdu_n}")
    st.sidebar.caption(f"Artifact: `{artifact.name}` · {stat.st_size // 1024} KB")
    if st.sidebar.button("Reload model cache"):
        _cached_model.clear()
        st.rerun()
    metrics = _load_metrics()
    if metrics:
        st.sidebar.write(f"**Test accuracy:** {metrics.get('accuracy', 0):.3f}")
        st.sidebar.write(f"**Macro F1:** {metrics.get('f1_macro', 0):.3f}")
    return payload


def page_single(payload: dict) -> None:
    st.header("Single review prediction")
    st.write(
        "Enter an English, Roman Urdu, Urdu-script, or code-mixed product review. "
        "Labels are **positive**, **negative**, or **neutral**, driven by polarity "
        "cues and word intensity (e.g. *bilkul bekaar* / *بہت اچھی*). "
        "Predictions are heuristic classifications, not definitive readings of intent."
    )
    text = st.text_area(
        "Review text",
        height=140,
        placeholder="e.g. Quality achi hai but delivery late thi",
        key="single_review_input",
    )
    examples = {
        "English positive": "Excellent product quality and fast delivery.",
        "Roman Urdu negative": "Product acha nahi hai bilkul bekaar.",
        "Neutral (mixed)": "Product theek hai na bohat acha na bura.",
        "Code-mixed contrast": "Product acha hai lekin quality kharab hai",
        "Urdu positive": "یہ پروڈکٹ بہت اچھی ہے، کوالٹی زبردست ہے۔",
        "Urdu negative": "یہ پروڈکٹ بالکل بیکار ہے، ایک دن میں ٹوٹ گیا۔",
        "Urdu neutral": "پروڈکٹ ٹھیک ہے، نہ بہت اچھا نہ برا۔",
    }
    items = list(examples.items())
    for row_start in range(0, len(items), 4):
        chunk = items[row_start : row_start + 4]
        cols = st.columns(len(chunk))
        for col, (label, sample) in zip(cols, chunk):
            if col.button(label, key=f"ex_{label}"):
                st.session_state["single_review_input"] = sample
                st.rerun()

    if st.button("Analyze sentiment", type="primary"):
        try:
            result = predict_one(text, payload=payload)
            st.session_state["single_result"] = result
            st.session_state["single_result_csv"] = dataframe_to_csv_bytes(
                format_single_result_csv(result)
            )
            # Rerun so download widgets are not created on the Analyze click
            # (helps avoid IDM auto-intercept). Downloads still show automatically.
            st.rerun()
        except PredictionError as exc:
            st.session_state.pop("single_result", None)
            st.session_state.pop("single_result_csv", None)
            st.warning(str(exc))
            return

    result = st.session_state.get("single_result")
    if not result:
        return

    sentiment = result["prediction"]
    if sentiment == "positive":
        st.success(f"Predicted sentiment: **{sentiment}**")
    elif sentiment == "neutral":
        st.info(
            f"Predicted sentiment: **{sentiment}** "
            "(mixed / factual tone, or low model confidence)"
        )
    else:
        st.error(f"Predicted sentiment: **{sentiment}**")

    st.write(f"**Model:** {result['model_name']}")
    if result["model_score"] is not None:
        st.write(
            f"**Model score:** `{result['model_score']:.4f}` — {result['score_note']}"
        )
    with st.expander("Cleaned text"):
        st.code(result["cleaned_text"])
    st.caption(result["limitations"])

    st.subheader("Download")
    st.caption("Click the button below when you want the Excel-friendly CSV.")
    st.download_button(
        "Download result CSV",
        data=st.session_state["single_result_csv"],
        file_name="single_sentiment_result.csv",
        mime="text/csv",
        key="dl_single_result_csv",
    )


def page_batch(payload: dict) -> None:
    st.header("Batch review analysis")
    st.write(
        "Upload **any org table** with a review-text column "
        "(CSV / TSV / Excel / JSON). Column names do not need to match our "
        "curated schema — map them below. Files are data only, never executed. "
        f"Max {MAX_BATCH_ROWS} rows / {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
    )
    uploaded = st.file_uploader(
        "Data file",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl"],
    )
    if uploaded is None:
        return

    if uploaded.size is not None and uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File is too large. Please upload a smaller file.")
        return

    # Reset run lock when a different file is uploaded
    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("batch_file_key") != file_key:
        st.session_state["batch_file_key"] = file_key
        st.session_state["batch_run_disabled"] = False
        st.session_state["batch_pending"] = False
        for key in (
            "batch_result",
            "batch_meta",
            "batch_formatted",
            "batch_summary",
            "batch_text_col",
            "batch_zip",
            "batch_eval_metrics",
            "batch_ingest_report",
        ):
            st.session_state.pop(key, None)

    try:
        raw = uploaded.read()
        df, file_meta = read_tabular_bytes(raw, filename=uploaded.name)
    except IngestError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse file: {exc}")
        return

    if df.empty:
        st.error("The uploaded file has no rows.")
        return

    detected = detect_column_map(df)
    cols = list(df.columns)
    text_default = cols.index(detected.text) if detected.text in cols else 0
    label_options = ["(none — predict only)"] + cols
    label_default = 0
    if detected.label and detected.label in cols:
        label_default = label_options.index(detected.label)

    run_disabled = bool(
        st.session_state.get("batch_run_disabled")
        or st.session_state.get("batch_pending")
    )
    text_col = st.selectbox(
        "Text column (review body)",
        options=cols,
        index=text_default,
        disabled=run_disabled,
        key="batch_text_col_select",
    )
    label_choice = st.selectbox(
        "Optional gold label column (for evaluation only)",
        options=label_options,
        index=label_default,
        disabled=run_disabled,
        key="batch_label_col_select",
    )
    label_col = None if label_choice.startswith("(none") else label_choice

    try:
        _, cmap, report = prepare_for_prediction(
            df,
            text_column=text_col,
            label_column=label_col,
            encoding=file_meta.get("encoding"),
            fmt=str(file_meta.get("format", "csv")),
        )
    except IngestError as exc:
        st.error(str(exc))
        return

    if not run_disabled:
        st.session_state["batch_selected_text_col"] = cmap.text
        st.session_state["batch_selected_label_col"] = cmap.label
        st.session_state["batch_ingest_report"] = report.to_dict()
    text_col = st.session_state.get("batch_selected_text_col", cmap.text)

    with st.expander("Ingest check (format / columns)", expanded=True):
        st.write(
            f"**Format:** `{report.format}` · **Encoding:** `{report.encoding}` · "
            f"**Rows:** {report.n_rows} · **Empty text:** {report.empty_text_rows}"
        )
        st.write(
            f"**Text column:** `{report.text_column}`"
            + (
                f" · **Label column:** `{report.label_column}`"
                if report.label_column
                else " · **Label column:** none"
            )
        )
        if report.label_counts:
            st.write("Mapped label counts:", report.label_counts)
        for warn in report.warnings:
            st.warning(warn)
        sample_vals = df[text_col].astype(str).head(3).tolist()
        st.caption("Preview: " + " | ".join(sample_vals))

    if st.button(
        "Run batch prediction",
        type="primary",
        disabled=run_disabled,
        key="batch_run_button",
    ):
        st.session_state["batch_run_disabled"] = True
        st.session_state["batch_pending"] = True
        st.session_state["batch_selected_text_col"] = text_col
        st.session_state["batch_selected_label_col"] = label_col
        st.rerun()

    if st.session_state.get("batch_pending"):
        selected = st.session_state.get("batch_selected_text_col", text_col)
        try:
            if len(df) > MAX_BATCH_ROWS:
                raise PredictionError(
                    f"Too many rows ({len(df)}). Maximum is {MAX_BATCH_ROWS}."
                )
            with st.spinner("Running batch prediction…"):
                result = predict_dataframe(df, selected, payload=payload)
                _, meta = build_results_zip(result, selected)
            st.session_state["batch_result"] = result
            st.session_state["batch_text_col"] = selected
            st.session_state.pop("batch_zip", None)
            st.session_state["batch_meta"] = {
                "predictions_csv": meta["predictions_csv"],
                "summary_csv": meta["summary_csv"],
                "chart_png": meta["chart_png"],
            }
            st.session_state["batch_formatted"] = meta["formatted_df"]
            st.session_state["batch_summary"] = meta["summary_df"]
            batch_eval = _metrics_from_batch_result(result)
            if batch_eval is not None:
                st.session_state["batch_eval_metrics"] = batch_eval
            else:
                st.session_state.pop("batch_eval_metrics", None)
            st.session_state["batch_pending"] = False
            st.rerun()
        except PredictionError as exc:
            st.session_state["batch_run_disabled"] = False
            st.session_state["batch_pending"] = False
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.session_state["batch_run_disabled"] = False
            st.session_state["batch_pending"] = False
            st.error(f"Batch prediction failed: {exc}")
            return

    if "batch_result" not in st.session_state:
        return

    result = st.session_state["batch_result"]
    formatted = st.session_state.get("batch_formatted", result)
    summary = st.session_state.get(
        "batch_summary", sentiment_summary(result["prediction"])
    )
    meta = st.session_state.get("batch_meta", {})

    st.subheader("Predictions")
    st.dataframe(formatted, use_container_width=True)

    st.subheader("Sentiment distribution")
    if summary.empty:
        st.info("No valid sentiment predictions to summarize.")
    else:
        st.dataframe(summary, use_container_width=True)
        fig = make_distribution_figure(summary)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("Downloads")
    st.caption(
        "Download buttons appear after prediction. "
        "A file only downloads when you click one of them."
    )

    c1, c2, c3 = st.columns(3)
    if meta.get("predictions_csv"):
        c1.download_button(
            "Predictions CSV",
            data=meta["predictions_csv"],
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            key="dl_predictions_csv",
        )
    if meta.get("summary_csv"):
        c2.download_button(
            "Summary CSV",
            data=meta["summary_csv"],
            file_name="sentiment_summary.csv",
            mime="text/csv",
            key="dl_summary_csv",
        )
    if meta.get("chart_png"):
        c3.download_button(
            "Chart PNG",
            data=meta["chart_png"],
            file_name="sentiment_distribution.png",
            mime="image/png",
            key="dl_chart_png",
        )


def page_evaluation() -> None:
    st.header("Model evaluation")
    st.info(
        "Uploading a file in **Batch upload** does **not** change the training "
        "holdout metrics below. Those update only after you retrain "
        "(`python -m src.train` then `python -m src.evaluate`). If your batch "
        "file has gold labels (`gold_label` / `Sentiment` / `label` / stars), a "
        "**Latest batch evaluation** section appears and updates for that upload."
    )

    # --- Latest batch (only shown after a labelled Batch upload run) ---
    batch_metrics = st.session_state.get("batch_eval_metrics")
    if batch_metrics:
        _render_metrics_block(
            batch_metrics,
            title="Latest batch evaluation (updates with each labelled CSV run)",
            source_note=(
                f"Compared predictions to column `{batch_metrics.get('gold_column')}` "
                f"on {batch_metrics.get('n_compared')} rows from your last batch run."
            ),
        )
        st.divider()

    # --- Training holdout (static until retrain) ---
    metrics = _load_metrics()
    if not metrics:
        st.info(
            "No training evaluation report yet. After training, run:\n\n"
            "`python -m src.evaluate`"
        )
        return

    _render_metrics_block(
        metrics,
        title="Training holdout evaluation (from last model training)",
        source_note=(
            f"{metrics.get('model_name', 'Model')} · "
            f"train `{metrics.get('train_size', '—')}` · "
            f"test `{metrics.get('test_size', '—')}` · "
            f"baseline `{float(metrics.get('majority_baseline_accuracy', 0) or 0):.3f}`"
        ),
    )

    st.divider()
    st.subheader("Training data label balance")
    class_counts = metrics.get("class_counts_full") or {}
    if class_counts:
        counts_df = pd.DataFrame(
            [
                {"Sentiment": str(k).title(), "Count": int(v)}
                for k, v in class_counts.items()
            ]
        )
        st.dataframe(counts_df, use_container_width=True, hide_index=True)
        fig_c, ax_c = plt.subplots(figsize=(10, 3.2))
        bar_colors = []
        for s in counts_df["Sentiment"]:
            key = str(s).lower()
            if "negative" in key:
                bar_colors.append("#c62828")
            elif "neutral" in key:
                bar_colors.append("#546e7a")
            else:
                bar_colors.append("#2e7d32")
        y_pos = list(range(len(counts_df)))
        vals = counts_df["Count"].tolist()
        ax_c.barh(y_pos, vals, color=bar_colors, height=0.55)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(counts_df["Sentiment"].tolist())
        ax_c.invert_yaxis()
        max_val = max(vals) if vals else 1
        ax_c.set_xlim(0, max_val * 1.28)
        ax_c.set_xlabel("Number of reviews")
        ax_c.set_title("Label balance in full training dataset")
        for i, val in enumerate(vals):
            ax_c.text(
                val + max_val * 0.02,
                i,
                f"{val:,}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        for spine in ("top", "right"):
            ax_c.spines[spine].set_visible(False)
        ax_c.grid(axis="x", linestyle="--", alpha=0.35)
        ax_c.set_axisbelow(True)
        fig_c.tight_layout()
        st.pyplot(fig_c, clear_figure=True, use_container_width=True)
        plt.close(fig_c)

    st.caption(metrics.get("notes", ""))


def page_research() -> None:
    st.header("Research notes")
    st.caption(
        "Graduate-admissions-oriented notes linking English Linguistics "
        "(negation, contrast, variation, contact) to a transparent CPU-only NLP pipeline."
    )

    st.markdown(
        """
### Research question

> How do **negation**, **code-switching**, and **informal Roman Urdu spelling**
> affect the performance of sentiment classification models on bilingual
> **product reviews**?

**Why this question?** E-commerce reviews in South Asian markets are not clean
English. Shoppers write Roman Urdu, switch languages mid-sentence, negate with
unstable spellings, and place the real judgment after *but* / *lekin*. A model
that only “works on English adjectives” is not answering the linguistic problem.

**Disciplinary bridge**
- **English Linguistics:** polarity, negation scope, contrastive discourse,
  orthographic variation, language contact / code-switching
- **Computational methods:** bag-of-n-grams features, linear classifiers,
  leakage-safe evaluation, qualitative error analysis on linguistic probes
"""
    )

    with st.expander("1. What is sentiment analysis in this project?", expanded=True):
        st.markdown(
            """
Sentiment analysis here is a **supervised text classification** task:
map a review string to **positive**, **negative**, or **neutral**
(neutral covers mixed/factual tone and, on older binary models, low-confidence cases).

#### What the label means
The label is a **corpus annotation convention**, not a claim about the writer’s
inner state. Two humans can disagree on mixed reviews
(*I like the design but I am not happy with the battery*). We treat disagreement
as a known limit of annotation, not as “model failure” alone.

#### What this project is *not*
- Not full semantic parsing or dialogue understanding
- Not emotion recognition / psychology measurement
- Not sarcasm detection as a dedicated task

#### Neutral and Urdu script
Curated Neutral rows follow a simple guide: average, factual, or mixed with no
clear praise/complaint. **Urdu-script (نستعلیق)** letters are preserved in
preprocessing so those reviews are not wiped empty. Optional external
reference scripts may still drop Neutral when building binary-only test CSVs.
"""
        )

    with st.expander("2. Why bilingual / Roman Urdu / code-mixed reviews?", expanded=True):
        st.markdown(
            """
#### Three language situations in one marketplace genre

| Situation | Example | Linguistic issue |
|-----------|---------|------------------|
| English | *Excellent quality and fast delivery* | Familiar polarity lexicon |
| Roman Urdu | *Product bohat acha hai* | Urdu morphology/lexicon in Latin script |
| Code-mixed | *Quality achi hai but delivery late thi* | Contact: two systems in one utterance |

#### Why English-only tools struggle
Tokenizers and lexicons trained on standard English see Roman Urdu forms as rare
or OOV. Intensifiers (*bohat*), negation (*nahi*), and contrast (*lekin*) may be
down-weighted or deleted. From a linguistics view, that discarded material is
**evidence of variation and contact** — exactly what we want to keep visible.

#### Product-review genre matters
Reviews are short, evaluative, often clause-chained (*X was fine but Y failed*).
That makes **negation** and **contrast** unusually frequent relative to news text,
and makes bag-of-n-grams a reasonable first computational baseline for studying them.
"""
        )

    with st.expander("3. Negation — polarity flip (core of the research question)"):
        st.markdown(
            """
#### Linguistic idea
Negation can reverse the polarity contribution of nearby evaluative words
(*good* vs *not good*; *acha* vs *acha nahi*). Scope can be local
(*not good*) or interact with nouns (*no complaints* → overall positive).

#### Minimal pairs we care about

| Control (positive cue) | Negated form | Expected overall |
|------------------------|--------------|------------------|
| *This product is good* | *This product is **not** good* | negative |
| *Product acha hai* | *Product acha **nahi** hai* | negative |
| *Product acha hai* | *Product acha **nahin** hai* (spelling variant) | negative |
| complaint noun | ***No** complaints everything works well* | positive |

#### Computational design that protects the phenomenon
- **Preserve** English: *not*, *no*
- **Preserve** Roman Urdu: *nahi* (plus closed-set spelling map), *mat*, *na*
- **Do not** apply a large stopword purge that deletes these cues

#### Contrast with a common baseline mistake
Some Roman Urdu tutorials remove *nahi*, *nahin*, *na*, *mat*, and even *lekin*
as “stopwords.” That improves cleanliness for topic classification but **destroys**
the negation/contrast signal this research question needs. We explicitly reject
that policy (documented in `docs/REFERENCE_ANALYSIS.md` and
`docs/LINGUISTIC_DECISIONS.md`).
"""
        )

    with st.expander("4. Contrast — when the second clause carries the verdict"):
        st.markdown(
            """
#### Linguistic idea
Contrastive connectives (*but*, *lekin*, *magar*, *however*, *though*) often mark
that the **speaker’s final stance** is in the second clause, even if the first
clause contains positive adjectives.

| Example | Surface positives | Verdict clause | Expected |
|---------|-------------------|----------------|----------|
| *Product acha hai **lekin** quality kharab hai* | *acha* | *quality kharab* | negative |
| *Quality achi hai **but** delivery late thi* | *achi* | *delivery late* | negative |
| *Looks cheap **but** feels premium* | *cheap* | *feels premium* | positive |

#### Why bag-of-words alone is fragile here
A plain unigram model can overweight *acha* / *achi* and ignore that contrast
reassigns the overall judgment. That is a discourse-sensitive problem; classical
TF-IDF does not “parse” clauses.

#### What we do instead (light, transparent feature)
After cleaning, if a contrast marker appears, preprocessing:
1. keeps the pre-contrast setup once
2. inserts a `__contrast__` marker
3. **repeats the post-contrast clause** so TF-IDF can weight the verdict side more

This is **deterministic feature engineering**, not attention, dependency parsing,
or a claim that the model understands discourse structure. It is an explicit,
interview-defensible compromise for CPU-only n-gram models.
"""
        )

    with st.expander("5. Informal Roman Urdu spelling (orthographic variation)"):
        st.markdown(
            """
#### Linguistic idea
Roman Urdu has **no single standard orthography**. Writers improvise Latin-script
spellings for Urdu words. The same negation lemma can appear as *nahi*, *nahin*,
*nae*, *nhi*, etc. Product words compress (*qlty*, *zabrdast*).

#### Two bad extremes
1. **Do nothing:** every spelling is a separate sparse feature → fragile negation
2. **Aggressive phonetic normalization:** invent false equivalences; erase variation
   you wanted to study

#### Middle path used here (closed-set map)
Map only a documented handful of *nahi* variants → canonical `nahi`:

`nahin`, `naheen`, `nahe`, `nae`, `nai`, `naie`, `nhi`, `nhin`

Leave other informal spellings untouched so error analysis can still show sparsity
failures. Spelling variation remains part of the research object, not something we
pretend does not exist.
"""
        )

    with st.expander("6. Code-switching and language contact"):
        st.markdown(
            """
#### Linguistic idea
Code-switching / code-mixing is a **contact** phenomenon: speakers draw on two
lexicons and grammars in one utterance. In reviews this often appears as:
- English noun + Roman Urdu predicate (*Quality achi hai*)
- Roman Urdu clause + English contrast (*… but delivery late*)
- Borrowed evaluative English (*worst*, *recommended*) inside Urdu frames

#### Methodological rule
Keep mixed reviews as **single strings**. **Do not translate** Urdu spans to English
before classification.

If we translated first, we would classify an English paraphrase and quietly change
the object of study. Contact cues (switch points, mixed bigrams) would disappear
from the feature space.

#### Evaluation link
When language tags exist (`english` / `roman_urdu` / `code_mixed`), held-out
metrics are also reported **per language slice**, so we can discuss whether errors
concentrate in mixed or Roman Urdu text — a linguistics-relevant breakdown, not
only a single global accuracy number.
"""
        )

    with st.expander("7. Intensifiers and mixed / ambiguous sentiment"):
        st.markdown(
            """
#### Intensifiers
Forms like *bohat* / *very* strengthen polarity (*bohat acha* vs *bohat bura*).
We keep them; they are polarity-relevant morphology/lexicon, not stopwords.

#### Mixed sentiment
Real reviews often praise one aspect and condemn another. Binary labels force a
single choice. Our linguistic probes include an ambiguous case
(*I like the design but I am not happy with the battery*) annotated toward the
dissatisfaction on a key feature. Humans can disagree; the probe documents a
**research judgment**, not absolute truth.
"""
        )

    with st.expander("8. Computational method (transparent baseline)"):
        st.markdown(
            """
#### Pipeline (train = inference)
1. `clean_text` — shared preprocessing inside the sklearn Pipeline
2. **TF-IDF** unigrams + bigrams (fit **only** on the training split)
3. Linear model (**LinearSVC** primary; LogReg comparison available)
4. `class_weight="balanced"` for skewed label counts
5. **Sigmoid calibration** (`CalibratedClassifierCV`) for approximate probabilities
6. Optional `--tune`: stratified GridSearchCV on macro F1 (`C`, `max_features`, n-grams)

#### Why this baseline (for a linguistics + NLP interview)
| Goal | How this method helps |
|------|------------------------|
| Transparency | Features are n-grams you can inspect |
| Reproducibility | Seeded splits, saved holdout, metrics written from predictions |
| CPU / offline | No GPU, no commercial API, no LLM required to predict |
| Link errors to linguistics | Failures map to negation / contrast / spelling probes |

#### Leakage controls
- Stratified train/test split **before** vectorizer fit
- Exact-text dedupe
- `assert_no_text_leakage` between splits
- App uses the **same** preprocessor as training (no train/serve skew)

#### What calibration means (and does not mean)
UI scores are **approximate class probabilities** after sigmoid calibration.
They are still not a guarantee of correctness and not a measure of “confidence
in understanding meaning.”
"""
        )

    with st.expander("9. What we clean vs what we deliberately keep"):
        st.markdown(
            """
| Action | Examples | Rationale |
|--------|----------|-----------|
| Lowercase | *Acha* → *acha* | Reduce Latin-script sparsity |
| Strip URLs / emails / @mentions | links, handles | Marketplace noise, not polarity |
| Keep hashtag words | `#deal` → `deal` | May carry product cues |
| Strip most punctuation | `!!!` | Little help for bag-of-n-grams |
| **Keep negation** | *not*, *nahi*, … | Research core |
| **Keep contrast** | *but*, *lekin*, … | Verdict often in clause 2 |
| **Keep intensifiers** | *bohat*, *very* | Strengthens polarity |
| Closed-set *nahi* map | *nahin* → *nahi* | Targeted spelling robustness |
| Contrast emphasis | repeat post-*but* clause | Help n-grams see the verdict |

Full decision log: `docs/LINGUISTIC_DECISIONS.md`.
"""
        )

    with st.expander("10. Data strategy and licensing honesty"):
        st.markdown(
            """
| Layer | Role |
|-------|------|
| Curated MIT CSV (`data/raw/bilingual_product_reviews.csv`) | Small, redistributable, language-tagged seed set |
| Expanded training set (local) | Built via `scripts/build_expanded_dataset.py` |
| Open UCI-style sources (optional) | Research-use English / Roman Urdu growth when present |
| Local Daraz references (optional) | Experiments only — **not** assumed redistributable |
| Linguistic fixtures | Hand-crafted probes — not training targets to overfit |

**Honesty rule:** do not publish third-party review dumps unless licenses are clear.
Reference repos were inspected for **ideas only**; their metrics are **theirs**,
never claimed as this project’s results (`docs/REFERENCE_ANALYSIS.md`).
"""
        )

    with st.expander("11. Evaluation design (quantitative + qualitative)"):
        st.markdown(
            """
#### Quantitative (held-out)
- Stratified holdout after training
- Report **accuracy**, **macro precision / recall / F1**, **PR-AUC (positive)**
- Compare against a **majority-class baseline** (so imbalance cannot fake success)
- Optional **per-language** macro F1 when language tags exist
- Optional LogReg vs LinearSVC comparison on the **same** holdout

#### Qualitative (linguistic probes)
Hand-crafted examples in `tests/fixtures/linguistic_examples.csv` cover:
negation controls, English/Roman Urdu negation, spelling variants, *lekin* /
*but* contrast, intensifiers, informal spelling, mixed ambiguity, negation of
a complaint, and positive/negative code-switching.

Running `python -m src.linguistic_analysis` writes
`reports/linguistic_error_analysis.md`.

**Critical research stance:** probes are for **error analysis and discussion**.
We do **not** force the classifier to pass them by hard-coding rules into the
label. A miss is evidence for the research question, not something to hide.
"""
        )

    probe_path = ROOT_DIR / "tests" / "fixtures" / "linguistic_examples.csv"
    if probe_path.exists():
        with st.expander("12. Linguistic probe examples (fixture table)", expanded=True):
            st.caption(
                "Expected labels are linguistic annotations for analysis — "
                "not guaranteed model outputs."
            )
            try:
                probe_df = pd.read_csv(probe_path, encoding="utf-8")
                show = probe_df.rename(
                    columns={
                        "text": "Example",
                        "category": "Category",
                        "expected_sentiment": "Expected",
                        "notes": "Notes",
                    }
                )
                st.dataframe(show, use_container_width=True, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"Could not load probe fixture: {exc}")

    with st.expander("13. What the model can and cannot do"):
        st.markdown(
            """
#### Can
- Learn n-gram patterns associated with positive / negative reviews
- Run fully offline on CPU
- Show approximate calibrated probabilities in the UI
- Surface failures on negation / contrast / spelling probes for discussion
- Report per-language and baseline-aware metrics

#### Cannot
- Reliably model sarcasm, irony, or long discourse structure
- Guarantee every informal spelling
- Recover true speaker emotion or intent
- Replace human annotation guidelines for Neutral / mixed cases
- Turn high holdout scores into a claim of “understanding meaning”

**Interpretive limit:** model output is a **label**, not ground-truth meaning.
"""
        )

    with st.expander("14. Linguistics ↔ computation map (interview cheat-sheet)"):
        st.markdown(
            """
| Linguistics focus | What we observe in reviews | What the code does |
|-------------------|----------------------------|--------------------|
| Negation / polarity | *not*, *nahi*, *no complaints* | Preserve cues; probe minimal pairs |
| Contrastive discourse | *but*, *lekin* verdict shift | Preserve markers; emphasize clause 2 |
| Orthographic variation | *nahin* / *nae* / *qlty* | Closed-set *nahi* map; leave other variants |
| Language contact | Code-mixed clauses | No translation; mixed strings; language slices |
| Intensification | *bohat*, *very* | Keep intensifiers |
| Responsible science | Small / skewed data risks | Baseline, macro F1, PR-AUC, probe report |

#### Suggested oral narrative (60–90 seconds)
1. State the research question (negation, code-switching, spelling on bilingual reviews).
2. Explain why stopword deletion of *nahi* / *lekin* would be linguistically wrong.
3. Describe the transparent TF-IDF + linear baseline and leakage controls.
4. Show a probe example (e.g. *acha hai lekin quality kharab*) and how evaluation is honest.
5. End with limits: labels ≠ emotions; bag-of-n-grams ≠ discourse understanding.
"""
        )

    with st.expander("15. Where to read more in this repository"):
        st.markdown(
            """
| Document | Contents |
|----------|----------|
| `docs/LINGUISTIC_DECISIONS.md` | Why we keep negation/contrast; spelling map |
| `docs/PROJECT_PLAN.md` | Scope, milestones, interview story |
| `docs/REFERENCE_ANALYSIS.md` | What we learned / rejected from reference repos |
| `docs/TEST_REPORT.md` | Test evidence |
| `reports/evaluation_report.md` | Latest held-out metrics |
| `reports/linguistic_error_analysis.md` | Latest probe results |
| `reports/model_comparison.md` | LogReg vs LinearSVC on the same holdout |
| `data/README.md` | Expanded dataset build recipe / licensing notes |
"""
        )


def main() -> None:
    payload = sidebar_status()
    page = st.session_state.get("nav_page", _NAV_PAGES[0])

    if page == "Single review":
        if payload is None:
            st.warning("Train and place the model artifact before predicting.")
        else:
            page_single(payload)
    elif page == "Batch upload":
        if payload is None:
            st.warning("Train and place the model artifact before batch analysis.")
        else:
            page_batch(payload)
    elif page == "Evaluation":
        page_evaluation()
    else:
        page_research()

    st.markdown("---")
    st.caption(f"Project root: `{ROOT_DIR.name}` · Artifact: `{MODEL_ARTIFACT_PATH.name}`")


if __name__ == "__main__":
    main()
