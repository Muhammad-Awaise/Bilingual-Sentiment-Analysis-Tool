"""
Run reproducible feature / preprocessing ablation suite.

Example (CPU-friendly subsample):

    python scripts/run_ablation.py --max-rows 8000

Full data (slow):

    python scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ablation import run_ablation_suite  # noqa: E402
from src.config import resolve_default_train_csv  # noqa: E402
from src.data_loader import load_reviews_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=8000,
        help="Stratified subsample size (default 8000; set 0 for full data)",
    )
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    path = Path(args.data) if args.data else resolve_default_train_csv()
    df = load_reviews_csv(path)
    max_rows = None if args.max_rows == 0 else args.max_rows
    payload = run_ablation_suite(
        df,
        calibrate=not args.no_calibrate,
        reports_dir=args.reports_dir,
        max_rows=max_rows,
    )
    print(json.dumps({"n_configs": len(payload.get("results") or []), **{k: payload[k] for k in ("created_utc", "n_used", "subsampled")}}, indent=2))
    for row in payload.get("results") or []:
        print(
            f"{row['configuration']}: acc={row['accuracy']:.4f} "
            f"macroF1={row['f1_macro']:.4f} neuF1={row.get('neutral_f1')} "
            f"time={row['training_time_seconds']}s"
        )


if __name__ == "__main__":
    main()
