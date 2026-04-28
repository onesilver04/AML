import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_Y_TEST = PROJECT_ROOT / "y_test.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "SHAP/test_confidence_by_sample_bad_positive.csv"
DEFAULT_OUTPUT = "SHAP/Task/Classified/correct_classified_test_samples.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract every correctly classified sample from the saved test set."
    )
    parser.add_argument("--y-test", type=Path, default=DEFAULT_Y_TEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_csv(path: Path, name: str) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    return pd.read_csv(path)


def main():
    args = parse_args()

    y_test = load_csv(args.y_test, "y_test")
    predictions = load_csv(args.predictions, "predictions")

    if "class" not in y_test.columns:
        raise ValueError("y_test must contain a 'class' column.")
    required_prediction_cols = {
        "sample_idx",
        "true_class",
        "predicted_class",
        "is_correct",
    }
    missing_cols = required_prediction_cols - set(predictions.columns)
    if missing_cols:
        raise ValueError(
            "predictions file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    y_test = y_test.copy()
    y_test.insert(0, "sample_idx", range(len(y_test)))
    y_test = y_test.rename(columns={"class": "saved_true_class"})

    classified = predictions[predictions["is_correct"] == True].copy()
    classified = classified.merge(y_test, on="sample_idx", how="left")

    if classified["saved_true_class"].isna().any():
        missing = classified.loc[
            classified["saved_true_class"].isna(), "sample_idx"
        ].tolist()
        raise ValueError(f"Some sample_idx values were not found in y_test: {missing}")

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(output, index=False)

    total = len(predictions)
    count = len(classified)
    rate = count / total * 100 if total else 0.0
    sample_indices = classified["sample_idx"].astype(int).tolist()

    print(f"Total test samples: {total}")
    print(f"Correctly classified samples: {count} ({rate:.2f}%)")
    print(f"Correctly classified sample_idx: {sample_indices}")
    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    main()
