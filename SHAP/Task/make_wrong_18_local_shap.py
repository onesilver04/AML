import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHAP_DIR = PROJECT_ROOT / "SHAP"
if str(SHAP_DIR) not in sys.path:
    sys.path.insert(0, str(SHAP_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import shap  # noqa: E402

from shap_rf_non_prefix import (  # noqa: E402
    DROP_COLUMNS,
    build_raw_shap_records,
    load_target,
    make_rf,
    resolve_output_path,
    save_shap_tuples_json,
    to_risk_label,
    tune_rf,
)


DEFAULT_INPUT = PROJECT_ROOT / "SHAP/Task/wrong_18.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SHAP/Task/wrong_18_local_shap"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create local SHAP JSON files for samples listed in wrong_18.csv, "
            "using the same top-3 raw-feature format as SHAP/120 Local Shap."
        )
    )
    parser.add_argument("--data-path", default="german21_ohe.csv")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--search-iter", type=int, default=32)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sample_indices(path: Path, max_samples: int) -> list[int]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    samples = pd.read_csv(path)
    if "sample_idx" not in samples.columns:
        raise ValueError("Input CSV must contain a 'sample_idx' column.")

    sample_indices_series = pd.to_numeric(samples["sample_idx"], errors="coerce")
    skipped_rows = int(sample_indices_series.isna().sum())
    if skipped_rows:
        print(f"Skipped non-numeric sample_idx rows: {skipped_rows}")

    sample_indices = sample_indices_series.dropna().astype(int).tolist()
    invalid = [idx for idx in sample_indices if idx < 0 or idx >= max_samples]
    if invalid:
        raise IndexError(
            f"Invalid sample_idx values: {invalid}. "
            f"Valid range is 0 ~ {max_samples - 1}."
        )

    return sample_indices


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve_output_path(args.data_path))
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = load_target(df)
    X = df.drop(columns=["class"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    sample_indices = load_sample_indices(args.input, len(X_test))

    tuned = tune_rf(X_train, y_train, args)

    rf_for_shap = make_rf(**tuned["params"])
    rf_for_shap.fit(X_train, y_train)

    raw_proba_test = rf_for_shap.predict_proba(X_test)[:, 1]

    explainer = shap.TreeExplainer(rf_for_shap)
    shap_exp = explainer(X_test)
    if len(shap_exp.values.shape) == 3:
        shap_exp = shap_exp[:, :, 1]

    sv = shap_exp.values
    feature_names = X_train.columns.tolist()

    for sample_idx in sample_indices:
        records = build_raw_shap_records(
            sv=sv,
            feature_names=feature_names,
            sample_idx=sample_idx,
            top_k=args.top_k,
        )

        raw_bad_proba = float(raw_proba_test[sample_idx])
        prediction_label = (
            "BAD CREDIT RISK"
            if raw_bad_proba >= tuned["threshold"]
            else "GOOD CREDIT RISK"
        )

        save_shap_tuples_json(
            records,
            sample_idx,
            true_label=to_risk_label(int(y_test.iloc[sample_idx])),
            prediction_label=prediction_label,
            predict_proba=raw_bad_proba,
            threshold=float(tuned["threshold"]),
            save_path=output_dir / f"shap_tuples_non_prefix_{sample_idx}.json",
        )

    print(f"Input CSV      : {resolve_path(args.input)}")
    print(f"Output dir     : {output_dir}")
    print(f"Sample count   : {len(sample_indices)}")
    print(f"Top features   : {args.top_k}")
    print(f"RF threshold   : {tuned['threshold']:.4f}")


if __name__ == "__main__":
    main()
