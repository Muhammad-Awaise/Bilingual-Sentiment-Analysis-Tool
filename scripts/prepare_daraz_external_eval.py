"""
Prepare a Daraz-only external evaluation CSV (never for training).

    python scripts/prepare_daraz_external_eval.py
    python scripts/prepare_daraz_external_eval.py --max-rows 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import HOLDOUT_SPLIT_PATH, REPORTS_DIR  # noqa: E402
from src.daraz_eval import prepare_daraz_external_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument(
        "--exclude-train-overlap",
        action="store_true",
        help="Drop texts that appear in reports/holdout_split.joblib train texts",
    )
    args = parser.parse_args()

    train_texts = None
    split = REPORTS_DIR / HOLDOUT_SPLIT_PATH.name
    if args.exclude_train_overlap and split.exists():
        import joblib

        data = joblib.load(split)
        train_texts = list(data.get("X_train") or [])

    _df, manifest = prepare_daraz_external_eval(
        include_neutral=not args.binary_only,
        max_rows=args.max_rows,
        train_texts_to_exclude=train_texts,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
