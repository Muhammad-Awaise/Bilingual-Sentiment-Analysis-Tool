"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
TESTS_DIR = ROOT_DIR / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
EXTERNAL_DATA_DIR = RAW_DATA_DIR / "external"

# Default curated dataset (redistributable, project-authored)
DEFAULT_RAW_CSV = RAW_DATA_DIR / "bilingual_product_reviews.csv"
DEFAULT_PROCESSED_CSV = PROCESSED_DATA_DIR / "reviews_clean.csv"
# Preferred training set when present (built by scripts/build_expanded_dataset.py)
EXPANDED_TRAIN_CSV = PROCESSED_DATA_DIR / "reviews_train_expanded.csv"

# Saved sklearn Pipeline (preprocessor steps live inside via FunctionTransformer)
MODEL_ARTIFACT_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "evaluation_metrics.json"
METRICS_MD_PATH = REPORTS_DIR / "evaluation_report.md"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "linguistic_error_analysis.md"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.md"
TRAIN_META_PATH = REPORTS_DIR / "train_meta.json"
HOLDOUT_SPLIT_PATH = REPORTS_DIR / "holdout_split.joblib"

# Labels — binary classification only for this deliverable
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
SUPPORTED_LABELS = frozenset({LABEL_POSITIVE, LABEL_NEGATIVE})
LABEL_ORDER = [LABEL_NEGATIVE, LABEL_POSITIVE]

# Column names used in curated / processed data
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
LANGUAGE_COLUMN = "language"  # english | roman_urdu | code_mixed | unknown

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 8000
NGRAM_RANGE = (1, 2)
CV_FOLDS = 3
MAX_BATCH_ROWS = 200000
MAX_UPLOAD_BYTES = 150 * 1024 * 1024  # 150 MB soft limit for Streamlit uploads

# Light hyperparameter grid (used only with --tune)
TUNE_MAX_FEATURES = (4000, 8000, 12000)
TUNE_NGRAM_RANGES = ((1, 1), (1, 2))
TUNE_C_VALUES = (0.5, 1.0, 2.0)

MODEL_NAME = "TF-IDF + LinearSVC"

LIMITATIONS_BLURB = (
    "This model uses bag-of-ngrams features. It does not truly understand meaning, "
    "sarcasm, or speaker intent. When calibration is enabled, scores are "
    "sigmoid-calibrated class probabilities — still approximate, not certainty. "
    "Negation, code-switching, and informal Roman Urdu spelling can still cause errors."
)


def resolve_default_train_csv() -> Path:
    """Prefer expanded training CSV when present; otherwise curated raw CSV."""
    if EXPANDED_TRAIN_CSV.exists():
        return EXPANDED_TRAIN_CSV
    return DEFAULT_RAW_CSV
