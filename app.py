"""
Bilingual Sentiment Analyzer — Streamlit entry point.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    LABEL_ORDER,
    MAX_BATCH_ROWS,
    MAX_UPLOAD_BYTES,
    METRICS_JSON_PATH,
    MODEL_ARTIFACT_PATH,
    REPORTS_DIR,
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
from src.ui import (
    inject_base_css,
    loading_placeholder,
    metric_row,
    page_header,
    shimmer_metrics,
)

st.set_page_config(
    page_title="Bilingual Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    # Empty menu items so toolbarMode=minimal can hide the ⋮ control (sibling).
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
inject_base_css()


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
        "row_number",
        "score_note",
        "limitations",
    }
    for name in LABEL_COLUMN_CANDIDATES:
        key = str(name).strip().lower()
        if key in lower_map and key not in skip:
            return lower_map[key]
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
    if result is None or result.empty:
        return None
    pred_col = next(
        (
            c
            for c in ("prediction", "predicted_sentiment")
            if c in result.columns
        ),
        None,
    )
    if pred_col is None:
        return None
    gold_col = _find_gold_column(result)
    if gold_col is None:
        return None

    working = result.copy()
    gold_vals = working[gold_col]
    if isinstance(gold_vals, pd.DataFrame):
        gold_vals = gold_vals.iloc[:, 0]
    pred_vals = working[pred_col]
    if isinstance(pred_vals, pd.DataFrame):
        pred_vals = pred_vals.iloc[:, -1]
    working["_gold"] = gold_vals.map(normalize_label)
    working["_pred"] = pred_vals.map(
        lambda v: str(v).strip().lower() if v is not None else None
    )
    working = working[working["_gold"].isin(LABEL_ORDER)]
    working = working[working["_pred"].isin(LABEL_ORDER)]
    if working.empty:
        return None

    metrics = compute_metrics(
        working["_gold"].tolist(),
        working["_pred"].tolist(),
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


def sidebar_model_status() -> dict | None:
    """Model block below native page nav (sibling multipage sidebar pattern)."""
    st.sidebar.divider()
    st.sidebar.caption("English · Roman Urdu · Code-mixed · Urdu-script reviews")
    st.sidebar.markdown("### Model")
    artifact = MODEL_ARTIFACT_PATH
    if not artifact.exists():
        st.sidebar.error(
            "Model artifact missing. Train locally with:\n\n"
            "python -m src.train\n\nthen python -m src.evaluate"
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
            "Pull latest models/sentiment_pipeline.joblib, then click Reload."
        )
        if st.sidebar.button("Reload model cache", type="primary", use_container_width=True):
            _cached_model.clear()
            st.rerun()
        return None

    st.sidebar.success("Model loaded (CPU, offline)")
    st.sidebar.write(f"**Model:** {payload.get('model_name', 'unknown')}")
    st.sidebar.write(f"**Train size:** {train_size:,}")
    st.sidebar.write(f"**Urdu-script rows in train:** {urdu_n:,}")
    st.sidebar.caption(f"Artifact: {artifact.name} · {stat.st_size // 1024} KB")
    if st.sidebar.button("Reload model cache", type="primary", use_container_width=True):
        _cached_model.clear()
        st.rerun()

    # Plain text — st.metric in narrow sidebar columns truncates to "0...."
    metrics = _load_metrics()
    if metrics:
        st.sidebar.write(f"**Test accuracy:** {float(metrics.get('accuracy', 0)):.3f}")
        st.sidebar.write(f"**Macro F1:** {float(metrics.get('f1_macro', 0)):.3f}")
    return payload


def _require_model() -> dict | None:
    payload = st.session_state.get("_model_payload")
    if payload is None:
        st.warning("Train and place the model artifact before predicting.")
    return payload


def _page_single_review() -> None:
    payload = _require_model()
    if payload is not None:
        page_single(payload)


def _page_batch_upload() -> None:
    payload = _require_model()
    if payload is not None:
        page_batch(payload)


def _page_evaluation() -> None:
    page_evaluation()


def _page_research() -> None:
    page_research()


def page_single(payload: dict) -> None:
    page_header(
        "Single review prediction",
        "English · Roman Urdu · Urdu-script · code-mixed product reviews. "
        "Labels are positive / negative / neutral — heuristic classifications, "
        "not readings of human intent.",
    )
    st.write(
        "Enter a review. Predictions use polarity cues and word intensity "
        "(e.g. *bilkul bekaar* / *بہت اچھی*)."
    )
    text = st.text_area(
        "Review text",
        height=140,
        placeholder="e.g. Quality achi hai but delivery late thi",
        key="single_review_input",
    )

    if st.button("Analyze sentiment", type="primary"):
        try:
            with loading_placeholder(
                metrics=0, blocks=1, spinner="Analyzing sentiment…"
            ):
                result = predict_one(text, payload=payload)
                csv_bytes = dataframe_to_csv_bytes(format_single_result_csv(result))
            st.session_state["single_result"] = result
            st.session_state["single_result_csv"] = csv_bytes
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

    st.write(f"**Original review:** {result['text']}")
    st.write(f"**Model:** {result['model_name']}")
    if result["model_score"] is not None:
        label = "Calibrated confidence" if result.get("calibrated") else "Model score"
        st.write(
            f"**{label}:** `{result['model_score']:.4f}` — {result['score_note']}"
        )
    if result.get("low_confidence"):
        st.warning(
            "Low calibrated confidence — treat this prediction cautiously. "
            "Confidence is not certainty."
        )
    composition = result.get("language_composition") or {}
    if composition:
        st.write(
            f"**Detected composition:** `{composition.get('category', 'unknown')}` "
            f"(Latin={composition.get('has_latin')}, "
            f"Urdu-script={composition.get('has_urdu_script')}, "
            f"EN cues={composition.get('english_cue_hits', 0)}, "
            f"RU cues={composition.get('roman_urdu_cue_hits', 0)})"
        )
    if result.get("hybrid_heuristics_applied") and result.get(
        "raw_model_prediction"
    ) != result.get("prediction"):
        st.caption(
            f"Statistical model predicted `{result.get('raw_model_prediction')}`; "
            "a documented hybrid heuristic adjusted the displayed label."
        )
    with st.expander("Cleaned text (features seen by the vectorizer)"):
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
    page_header(
        "Batch review analysis",
        "Upload any org table (CSV / TSV / Excel / JSON). Map columns below — "
        f"max {MAX_BATCH_ROWS} rows / {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
        "Files are data only, never executed.",
    )
    st.write(
        "Column names do not need to match our curated schema. "
        "Prediction-only and labelled evaluation uploads are both supported."
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
            f"**Rows:** {report.n_rows} · **Empty text:** {report.empty_text_rows} · "
            f"**Duplicate texts:** {getattr(report, 'duplicate_text_rows', 0)}"
        )
        st.write(
            f"**Text column:** `{report.text_column}`"
            + (
                f" · **Label column:** `{report.label_column}`"
                if report.label_column
                else " · **Label column:** none (prediction-only)"
            )
        )
        st.caption(
            "Prediction-only uploads keep your original columns. "
            "Labelled uploads enable evaluation metrics. "
            "Training uses a separate normalized schema."
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
            with loading_placeholder(
                metrics=0, blocks=2, spinner="Running batch prediction…"
            ):
                result = predict_dataframe(df, selected, payload=payload)
                _, meta = build_results_zip(result, selected)
                batch_eval = _metrics_from_batch_result(result)
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
    page_header(
        "Model evaluation",
        "Held-out training metrics, optional latest batch evaluation, "
        "calibration / challenge / Daraz / ablation panels when reports exist.",
    )
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
    with loading_placeholder(
        metrics=4, blocks=2, spinner="Loading evaluation report…"
    ):
        metrics = _load_metrics()
        # Touch optional side reports so heavy JSON reads happen under shimmer
        daraz_raw = None
        ablation_raw = None
        daraz_path = REPORTS_DIR / "daraz_external_metrics.json"
        ablation_path = REPORTS_DIR / "ablation_results.json"
        if daraz_path.exists():
            try:
                daraz_raw = json.loads(daraz_path.read_text(encoding="utf-8"))
            except Exception:
                daraz_raw = None
        if ablation_path.exists():
            try:
                ablation_raw = json.loads(ablation_path.read_text(encoding="utf-8"))
            except Exception:
                ablation_raw = None

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

    # Neutral / language / source / calibration / challenge / Daraz / ablation
    neu = metrics.get("neutral_analysis") or {}
    if neu.get("neutral_present"):
        st.subheader("Neutral-class analysis")
        st.write(
            f"Support `{neu.get('support')}` · "
            f"P/R/F1 `{neu.get('precision'):.3f}` / "
            f"`{neu.get('recall'):.3f}` / `{neu.get('f1'):.3f}`"
        )
        st.caption(neu.get("notes", ""))
        c1, c2 = st.columns(2)
        with c1:
            st.write("Gold Neutral predicted as:", neu.get("true_neutral_predicted_as"))
        with c2:
            st.write("Predicted Neutral but true:", neu.get("predicted_neutral_but_true"))

    per_lang = metrics.get("per_language") or {}
    if per_lang:
        st.subheader("Per-language metrics")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Language": k,
                        "N": v.get("n_samples"),
                        "Accuracy": v.get("accuracy"),
                        "F1 macro": v.get("f1_macro"),
                    }
                    for k, v in per_lang.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    per_src = metrics.get("per_source") or {}
    if per_src:
        st.subheader("Per-source metrics")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Source": k,
                        "N": v.get("n_samples"),
                        "Accuracy": v.get("accuracy"),
                        "F1 macro": v.get("f1_macro"),
                        "Neutral F1": v.get("neutral_f1"),
                    }
                    for k, v in per_src.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    cal = metrics.get("calibration") or {}
    if cal.get("available"):
        st.subheader("Calibration analysis")
        st.write(
            f"Brier `{cal.get('brier_score'):.4f}` · "
            f"ECE `{cal.get('expected_calibration_error'):.4f}` · "
            f"Mean confidence `{cal.get('confidence_distribution', {}).get('mean', float('nan')):.4f}`"
        )
        st.caption(cal.get("notes", ""))
        rel = cal.get("reliability_diagram") or []
        if rel:
            rel_df = pd.DataFrame(rel)
            rel_df = rel_df[rel_df["count"] > 0]
            if not rel_df.empty:
                st.dataframe(rel_df, use_container_width=True, hide_index=True)

    challenge = metrics.get("challenge_set") or {}
    if challenge.get("available"):
        st.subheader("Linguistic challenge set")
        st.write(
            f"N `{challenge.get('n_examples')}` · "
            f"correct `{challenge.get('correct')}` · "
            f"incorrect `{challenge.get('incorrect')}` · "
            f"accuracy `{challenge.get('accuracy'):.3f}`"
        )
        st.caption(challenge.get("notes", ""))
        by_cat = challenge.get("by_category") or {}
        if by_cat:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Category": k,
                            "N": v.get("n"),
                            "Correct": v.get("correct"),
                            "Accuracy": v.get("accuracy"),
                            "F1 macro": v.get("f1_macro"),
                        }
                        for k, v in by_cat.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        errs = challenge.get("error_examples") or []
        if errs:
            with st.expander("Challenge-set error examples"):
                st.dataframe(pd.DataFrame(errs), use_container_width=True, hide_index=True)

    if daraz_raw:
        daraz = daraz_raw
        st.subheader("External Daraz performance")
        if daraz.get("usable"):
            st.write(
                f"N `{daraz.get('n_samples')}` · "
                f"accuracy `{daraz.get('accuracy'):.4f}` · "
                f"macro F1 `{daraz.get('f1_macro'):.4f}`"
            )
            st.write("Δ vs internal holdout:", daraz.get("delta_vs_internal_holdout"))
            for lim in daraz.get("limitations") or []:
                st.caption(lim)
        else:
            st.warning(
                "Daraz external eval not usable: "
                + "; ".join(daraz.get("limitations") or ["see report"])
            )

    if ablation_raw:
        ablation = ablation_raw
        st.subheader("Ablation comparisons")
        st.caption(
            f"Seed `{ablation.get('random_seed')}` · "
            f"rows used `{ablation.get('n_used')}` · "
            f"subsampled `{ablation.get('subsampled')}`"
        )
        rows = ablation.get("results") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    from src.reproducibility import stale_artifact_report

    stale = stale_artifact_report()
    if stale.get("warnings"):
        st.subheader("Artifact consistency")
        for w in stale["warnings"]:
            st.warning(w)

    st.caption(metrics.get("notes", ""))


def page_research() -> None:
    """Sibling-style Research Center + About brief layout for this project's notes."""
    page_header(
        "Research Center",
        "Local research workspace: linguistic framing, methodology, probes, and "
        "repository evidence — CPU-only, offline local models.",
    )

    st.markdown(
        """
**Purpose of this page**

Keep the scientific side of the project organized in one place — local tools only:

| Tab | What it’s for |
|---|---|
| **Overview** | Research question, label meaning, bilingual / script motivation |
| **Linguistics** | Negation, contrast, spelling, code-switching, intensifiers |
| **Methodology** | Pipeline, keep/drop decisions, data honesty, evaluation design |
| **Evidence** | Probe fixtures, limits, interview map, docs & reports |

This is **documentation + evidence**, not the predictor itself. Use **Single review** /
**Batch upload** for inference; use **Evaluation** for held-out metrics.
"""
    )

    skel = st.empty()
    with skel.container():
        shimmer_metrics(4)
    with st.spinner("Loading research snapshot…"):
        metrics = _load_metrics() or {}
        payload = st.session_state.get("_model_payload") or {}
        train_meta: dict = {}
        meta_path = REPORTS_DIR / "train_meta.json"
        if meta_path.exists():
            try:
                train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                train_meta = {}
    skel.empty()

    st.markdown(
        """
<div style="padding:1.25rem 1.5rem;border-radius:16px;color:#fff;
background:linear-gradient(135deg,#071526 0%,#0b1f3a 50%,#1e4fd6 140%);margin-bottom:1rem;">
  <div style="font-size:0.75rem;letter-spacing:0.18em;text-transform:uppercase;opacity:0.85;">
    Detailed research brief
  </div>
  <div style="font-size:1.75rem;font-weight:700;margin-top:0.35rem;">
    Bilingual Product-Review Sentiment Analyzer
  </div>
  <p style="margin:0.65rem 0 0;max-width:48rem;opacity:0.92;line-height:1.55;">
    A free, local research tool for English · Roman Urdu · code-mixed · Urdu-script
    product reviews — with transparent n-gram features, leakage-aware evaluation,
    and linguistic probes. No commercial cloud APIs required.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    acc = metrics.get("accuracy")
    f1 = metrics.get("f1_macro")
    metric_row(
        [
            (
                "Test accuracy",
                f"{float(acc):.3f}" if isinstance(acc, (int, float)) else "n/a",
            ),
            (
                "Macro F1",
                f"{float(f1):.3f}" if isinstance(f1, (int, float)) else "n/a",
            ),
            (
                "Train rows",
                f"{int(payload.get('train_size') or train_meta.get('n_records') or 0):,}"
                if (payload.get("train_size") or train_meta.get("n_records"))
                else "n/a",
            ),
            (
                "Active model",
                payload.get("model_name") or "Train artifact to load",
            ),
        ]
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.info("**Domain**\n\nBilingual product-review sentiment")
    k2.info("**Core task**\n\nPositive · negative · neutral classification")
    k3.info("**Output**\n\nLabels · scores · batch export · metrics")
    k4.info("**Constraint**\n\nFree · local · CPU · offline")

    c1, c2 = st.columns(2)
    lang_counts = payload.get("language_counts_full") or train_meta.get(
        "language_distribution"
    ) or {}
    urdu_n = int(lang_counts.get("urdu") or 0)
    c1.info(
        f"**Active model**\n\n{payload.get('model_name') or '—'} · "
        f"train size {int(payload.get('train_size') or 0):,}"
    )
    if isinstance(f1, (int, float)):
        c2.success(f"**Best held-out Macro F1**\n\n{float(f1):.3f}")
    else:
        c2.warning("**Best held-out Macro F1**\n\nNo evaluation metrics yet")
    if urdu_n:
        st.caption(f"Urdu-script rows represented in training snapshot: **{urdu_n:,}**")

    st.markdown("### Contents")
    st.caption("Open a tab below. Live metrics appear at the top when reports are present.")

    tab_overview, tab_ling, tab_method, tab_evidence = st.tabs(
        ["Overview", "Linguistics", "Methodology", "Evidence"]
    )

    with tab_overview:
        st.subheader("Research question")
        st.caption("What this workspace is defending in an interview or write-up.")
        with st.container(border=True):
            st.markdown(
                """
> How do **negation**, **code-switching**, **informal Roman Urdu spelling**,
> and **Urdu-script (نستعلیق) reviews** affect sentiment classification on
> bilingual **product reviews**?

**Why this question?** E-commerce reviews in South Asian markets are not clean
English. Shoppers write Roman Urdu, Urdu script, switch languages mid-sentence,
negate with unstable spellings, and place the real judgment after *but* / *lekin*.
A model that only “works on English adjectives” — or that collapses every
Nastaliq row to a constant prior — is not answering the linguistic problem.

**Disciplinary bridge**
- **English Linguistics:** polarity, negation scope, contrastive discourse,
  orthographic variation, language contact / code-switching, script choice
- **Computational methods:** word + character n-grams, linear classifiers,
  leakage-safe evaluation, flexible org-data ingest, qualitative probes
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

        with st.expander("2. Why bilingual / Roman Urdu / Urdu-script / code-mixed?", expanded=True):
            st.markdown(
                """
#### Four language situations in one marketplace genre

| Situation | Example | Linguistic issue |
|-----------|---------|------------------|
| English | *Excellent quality and fast delivery* | Familiar polarity lexicon |
| Roman Urdu | *Product bohat acha hai* | Urdu morphology/lexicon in Latin script |
| Urdu script | *یہ پروڈکٹ بہت اچھی ہے* | Same polarity cues in Arabic script (نستعلیق) |
| Code-mixed | *Quality achi hai but delivery late thi* | Contact: two systems in one utterance |

#### Why English-only / Latin-only tools struggle
Tokenizers and lexicons trained on standard English see Roman Urdu forms as rare
or OOV. Intensifiers (*bohat* / *بہت*), negation (*nahi* / *نہیں*), and contrast
(*lekin* / *لیکن*) may be down-weighted or deleted. If Arabic letters are stripped,
Urdu-script reviews become **empty vectors** and every row can collapse to the same
weak prior (historically a flat ~0.53 “negative”). From a linguistics view, that
discarded material is **evidence of variation, script choice, and contact** —
exactly what we want to keep visible.

#### Product-review genre matters
Reviews are short, evaluative, often clause-chained (*X was fine but Y failed*).
That makes **negation** and **contrast** unusually frequent relative to news text,
and makes n-gram baselines a reasonable first computational setup for studying them.
"""
            )

    with tab_ling:
        st.subheader("Linguistic phenomena")
        st.caption("Core constructs the preprocessor protects and the probes stress-test.")

        with st.expander("3. Negation — polarity flip (core of the research question)", expanded=True):
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
- **Preserve** Urdu script: *نہیں*, *نہ*, *مت* (and related polarity lexicon)
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
Contrastive connectives (*but*, *lekin*, *magar*, *however*, *though*,
*لیکن*, *مگر*) often mark that the **speaker’s final stance** is in the second
clause, even if the first clause contains positive adjectives.

| Example | Surface positives | Verdict clause | Expected |
|---------|-------------------|----------------|----------|
| *Product acha hai **lekin** quality kharab hai* | *acha* | *quality kharab* | negative |
| *Quality achi hai **but** delivery late thi* | *achi* | *delivery late* | negative |
| *اچھا ہے **لیکن** کوالٹی خراب ہے* | *اچھا* | *کوالٹی خراب* | negative |
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
When language tags exist (`english` / `roman_urdu` / `code_mixed` / `urdu`),
held-out metrics are also reported **per language slice**, so we can discuss
whether errors concentrate in mixed, Roman Urdu, or Urdu-script text — a
linguistics-relevant breakdown, not only a single global accuracy number.
"""
            )

        with st.expander("7. Intensifiers and mixed / ambiguous sentiment"):
            st.markdown(
                """
#### Intensifiers
Forms like *bohat* / *very* / *بہت* / *بالکل* strengthen polarity
(*bohat acha* vs *bohat bura*; *بہت اچھی* vs *بالکل بیکار*).
We keep them and lightly re-weight strong polarity cues; they are
polarity-relevant morphology/lexicon, not stopwords.

#### Mixed sentiment
Real reviews often praise one aspect and condemn another. Binary labels force a
single choice. Our linguistic probes include an ambiguous case
(*I like the design but I am not happy with the battery*) annotated toward the
dissatisfaction on a key feature. Humans can disagree; the probe documents a
**research judgment**, not absolute truth.
"""
            )

    with tab_method:
        st.subheader("Methodology (DRAFT framing)")
        st.warning(
            "Draft research framing tied to the shipped CPU pipeline — not fabricated results."
        )
        st.caption(
            "Fill Results / Discussion from Evaluation reports and probe runs. "
            "This tab documents method choices, not new metrics."
        )

        with st.expander("8. Computational method (transparent baseline)", expanded=True):
            st.markdown(
                """
#### Pipeline (train = inference)
1. `clean_text` — shared preprocessing inside the sklearn Pipeline
2. **FeatureUnion:** word TF-IDF (1–2 grams) **+** character TF-IDF (`char_wb` 3–5)
   so Urdu-script tokens stay informative even when rare / OOV
3. Linear model (**LinearSVC** primary, calibrated; LogReg comparison available)
4. `class_weight="balanced"` for skewed label counts
5. Large expanded train set + **Urdu-script corpus** so Nastaliq rows do not
   collapse to a constant prior
6. Optional `--tune`: stratified GridSearchCV on macro F1 (`C`, `max_features`, n-grams)

#### Why this baseline (for a linguistics + NLP interview)
| Goal | How this method helps |
|------|------------------------|
| Transparency | Features are n-grams you can inspect |
| Script robustness | Char n-grams + Urdu training rows keep Nastaliq alive |
| Reproducibility | Seeded splits, saved holdout, metrics written from predictions |
| CPU / offline | No GPU, no commercial API, no LLM required to predict |
| Link errors to linguistics | Failures map to negation / contrast / spelling / script probes |

#### Leakage controls
- Stratified train/test split **before** vectorizer fit
- Exact-text dedupe
- `assert_no_text_leakage` between splits
- App uses the **same** preprocessor as training (no train/serve skew)
- Model cache keyed by artifact mtime/size so a new `.joblib` reloads

#### What calibration means (and does not mean)
UI scores are **approximate class probabilities** after sigmoid calibration
(LinearSVC) or native LogReg probabilities. They are still not a guarantee of
correctness and not a measure of “confidence in understanding meaning.”
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
| Strip most punctuation | `!!!`، `۔` | Little help for bag-of-n-grams |
| **Keep Arabic / Urdu letters** | *اچھی*, *بیکار* | Nastaliq reviews must not wipe empty |
| **Keep negation** | *not*, *nahi*, *نہیں* | Research core |
| **Keep contrast** | *but*, *lekin*, *لیکن* | Verdict often in clause 2 |
| **Keep intensifiers** | *bohat*, *very*, *بہت* | Strengthens polarity |
| Closed-set *nahi* map | *nahin* → *nahi* | Targeted spelling robustness |
| Intensity markers | `__pos_intense__`, … | Light re-weight for strong polarity |
| Contrast emphasis | repeat post-*but* clause | Help n-grams see the verdict |

Full decision log: `docs/LINGUISTIC_DECISIONS.md`.
"""
            )

        with st.expander("10. Data strategy and licensing honesty"):
            st.markdown(
                """
| Layer | Role |
|-------|------|
| Curated MIT CSV (`data/raw/bilingual_product_reviews.csv`) | Small, redistributable, language-tagged seed (incl. Urdu script + Neutral) |
| Expanded training set (`reviews_train_expanded.csv`) | Large multi-source set via `scripts/build_expanded_dataset.py` (~76k rows) |
| Open / research CSVs | UCI Roman Urdu, Kaggle Roman Urdu, TweetEval, Amazon/IMDB/Yelp when present |
| Synthetic Urdu-script corpus | Project-authored Nastaliq templates so script coverage cannot vanish |
| Local Daraz references (optional) | Experiments only — **not** assumed redistributable |
| Flexible org ingest (`src/ingest.py`) | CSV / TSV / Excel / JSON; map *Review* / *Sentiment* / stars — orgs need not match our schema |
| Linguistic fixtures | Hand-crafted probes — not training targets to overfit |

**Honesty rule:** do not publish third-party review dumps unless licenses are clear.
Reference repos were inspected for **ideas only**; their metrics are **theirs**,
never claimed as this project’s results (`docs/REFERENCE_ANALYSIS.md`).
Binary external files (e.g. Kaggle Roman Urdu) have **no Neutral gold labels** by
construction — the model can still emit Neutral on mixed tone, but those CSVs
will not show Neutral as ground truth.
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
a complaint, and positive/negative code-switching. Urdu-script behaviour is
checked on the curated seed and holdout language slice (`urdu`).

Running `python -m src.linguistic_analysis` writes
`reports/linguistic_error_analysis.md`.

**Critical research stance:** probes are for **error analysis and discussion**.
We do **not** force the classifier to pass them by hard-coding rules into the
label. A miss is evidence for the research question, not something to hide.
"""
            )

    with tab_evidence:
        st.subheader("Evidence & limits")
        st.caption(
            "Stored fixtures and repo documents — not invented citations or fabricated scores."
        )

        probe_path = ROOT_DIR / "tests" / "fixtures" / "linguistic_examples.csv"
        with st.container(border=True):
            st.markdown("##### Linguistic probe examples")
            st.caption(
                "Expected labels are linguistic annotations for analysis — "
                "not guaranteed model outputs."
            )
            if probe_path.exists():
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
            else:
                st.info("Probe fixture not found at tests/fixtures/linguistic_examples.csv")

        with st.expander("13. What the model can and cannot do", expanded=True):
            st.markdown(
                """
#### Can
- Learn word/char n-gram patterns for positive / negative / neutral reviews
- Score English, Roman Urdu, code-mixed, and Urdu-script text offline on CPU
- Accept org tables via flexible column mapping (not only our curated schema)
- Show approximate probabilities in the UI
- Surface failures on negation / contrast / spelling / script probes
- Report per-language and baseline-aware metrics

#### Cannot
- Reliably model sarcasm, irony, or long discourse structure
- Guarantee every informal spelling or every unseen Nastaliq phrasing
- Recover true speaker emotion or intent
- Replace human annotation guidelines for Neutral / mixed cases
- Invent Neutral gold labels inside binary-only external CSVs
- Turn high holdout scores into a claim of “understanding meaning”

**Interpretive limit:** model output is a **label**, not ground-truth meaning.
"""
            )

        with st.expander("14. Linguistics ↔ computation map (interview cheat-sheet)"):
            st.markdown(
                """
| Linguistics focus | What we observe in reviews | What the code does |
|-------------------|----------------------------|--------------------|
| Negation / polarity | *not*, *nahi*, *نہیں*, *no complaints* | Preserve cues; probe minimal pairs |
| Contrastive discourse | *but*, *lekin*, *لیکن* verdict shift | Preserve markers; emphasize clause 2 |
| Orthographic variation | *nahin* / *nae* / *qlty* | Closed-set *nahi* map; leave other variants |
| Script choice | Urdu in Latin vs نستعلیق | Keep Arabic letters; char n-grams + Urdu train rows |
| Language contact | Code-mixed clauses | No translation; mixed strings; language slices |
| Intensification | *bohat*, *very*, *بہت* | Keep intensifiers; light intensity markers |
| Responsible science | Skewed / multi-source data | Baseline, macro F1, PR-AUC, probe report |

#### Suggested oral narrative (60–90 seconds)
1. State the research question (negation, code-switching, spelling, **and Urdu script**
   on bilingual product reviews).
2. Explain why deleting *nahi* / *lekin* — or stripping Arabic letters — would be
   linguistically wrong (and caused empty-vector collapse historically).
3. Describe word+char TF-IDF, LinearSVC, large expanded data, and leakage controls.
4. Show a probe (e.g. *acha hai lekin quality kharab* or an Urdu positive) and how
   evaluation stays honest; note org files map via ingest without forcing our schema.
5. End with limits: labels ≠ emotions; n-grams ≠ discourse understanding.
"""
            )

        with st.expander("15. Where to read more in this repository"):
            st.markdown(
                """
| Document | Contents |
|----------|----------|
| `docs/LINGUISTIC_DECISIONS.md` | Why we keep negation/contrast; spelling map; Urdu letters |
| `docs/PROJECT_PLAN.md` | Scope, milestones, interview story |
| `docs/REFERENCE_ANALYSIS.md` | What we learned / rejected from reference repos |
| `docs/TEST_REPORT.md` | Test evidence |
| `reports/evaluation_report.md` | Latest held-out metrics |
| `reports/linguistic_error_analysis.md` | Latest probe results |
| `reports/model_comparison.md` | LogReg vs LinearSVC on the same holdout |
| `data/README.md` | Expanded dataset build / org ingest / licensing notes |
| `src/ingest.py` | Flexible CSV / Excel / JSON column mapping for orgs |
"""
            )

        st.divider()
        st.markdown("##### Local report files")
        st.caption("Download stored Markdown reports when present (same honesty rule as sibling exports).")
        report_files = [
            ("Evaluation report", REPORTS_DIR / "evaluation_report.md"),
            ("Linguistic error analysis", REPORTS_DIR / "linguistic_error_analysis.md"),
            ("Model comparison", REPORTS_DIR / "model_comparison.md"),
            ("Ablation results", REPORTS_DIR / "ablation_results.md"),
            ("Daraz external report", REPORTS_DIR / "daraz_external_report.md"),
        ]
        for label, path in report_files:
            if path.exists():
                st.download_button(
                    f"Download {label}",
                    data=path.read_text(encoding="utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                    key=f"research_dl_{path.name}",
                )
            else:
                st.caption(f"{label}: not generated yet (`{path.name}`)")

    st.divider()
    st.caption(
        "Research Center = linguistic framing, methodology draft, probes, and repo evidence. "
        "Evaluation = quantitative held-out metrics. Predictor pages = inference only."
    )


def main() -> None:
    # Sibling-style page list in the sidebar (not radio buttons).
    nav_pages = [
        st.Page(_page_single_review, title="Single review", default=True),
        st.Page(_page_batch_upload, title="Batch upload"),
        st.Page(_page_evaluation, title="Evaluation"),
        st.Page(_page_research, title="Research"),
    ]
    try:
        pg = st.navigation(nav_pages, position="sidebar", expanded=True)
    except TypeError:
        # Older Streamlit without the expanded kwarg
        pg = st.navigation(nav_pages, position="sidebar")
    st.session_state["_model_payload"] = sidebar_model_status()
    pg.run()

    st.markdown("---")
    st.caption(f"Project root: `{ROOT_DIR.name}` · Artifact: `{MODEL_ARTIFACT_PATH.name}`")


if __name__ == "__main__":
    main()
