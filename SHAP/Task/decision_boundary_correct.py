import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "age",
    "existing_credits",
    "checking_status_0<=X<200",
    "checking_status_<0",
    "checking_status_>=200",
    "checking_status_no checking",
    "credit_history_all paid",
    "credit_history_critical/other existing credit",
    "credit_history_delayed previously",
    "credit_history_existing paid",
    "credit_history_no credits/all paid",
    "purpose_business",
    "purpose_domestic appliance",
    "purpose_education",
    "purpose_furniture/equipment",
    "purpose_new car",
    "purpose_other",
    "purpose_radio/tv",
    "purpose_repairs",
    "purpose_retraining",
    "purpose_used car",
    "savings_status_100<=X<500",
    "savings_status_500<=X<1000",
    "savings_status_<100",
    "savings_status_>=1000",
    "savings_status_no known savings",
    "employment_1<=X<4",
    "employment_4<=X<7",
    "employment_<1",
    "employment_>=7",
    "employment_unemployed",
    "housing_for free",
    "housing_own",
    "housing_rent",
    "job_high qualif/self emp/mgmt",
    "job_skilled",
    "job_unemp/unskilled non res",
    "job_unskilled resident",
    "Sex",
    "Married",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shap-dir",
        default="SHAP/Test Dataset Local Shap 25",
    )
    parser.add_argument(
        "--confidence-csv",
        default="SHAP/confidence.csv",
    )
    parser.add_argument(
        "--x-test",
        default="X_test.csv",
    )
    parser.add_argument(
        "--decision-boundary",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        default="SHAP/test_dataset_decision_boundary_features.csv",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def extract_sample_idx(path: Path) -> int:
    match = re.search(r"shap_tuples_non_prefix_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Invalid filename: {path.name}")
    return int(match.group(1))


def label_to_class(label: str) -> int:
    label = str(label).strip().upper()
    if label == "BAD CREDIT RISK":
        return 1
    if label == "GOOD CREDIT RISK":
        return 0
    raise ValueError(f"Unknown label: {label}")

def load_confidence_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"confidence CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        "sample_idx",
        "warning_type",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"confidence CSV missing columns: {sorted(missing)}"
        )

    return df[["sample_idx", "warning_type"]].copy()


def load_shap_jsons(shap_dir: Path) -> pd.DataFrame:
    json_paths = sorted(
        shap_dir.glob("shap_tuples_non_prefix_*.json"),
        key=extract_sample_idx,
    )

    if not json_paths:
        raise FileNotFoundError(
            f"No shap_tuples_non_prefix_*.json found in: {shap_dir}"
        )

    rows = []

    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_idx = int(data.get("sample_idx", extract_sample_idx(path)))

        true_label = data["true_label"]
        prediction = data["prediction"]
        predicted_label = prediction["predict_label"]

        confidence = float(prediction["calibrated_confidence"])

        true_class = label_to_class(true_label)
        predicted_class = label_to_class(predicted_label)

        # JSON에는 raw_bad_probability가 없으므로
        # calibrated_confidence에서 calibrated bad probability를 복원함.
        if predicted_class == 1:
            bad_probability = confidence
        else:
            bad_probability = 1.0 - confidence

        good_probability = 1.0 - bad_probability

        rows.append(
            {
                "sample_idx": sample_idx,
                "true_class": true_class,
                "true_label": true_label,
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "is_correct": true_class == predicted_class,
                "probability_raw_bad_probability": bad_probability,
                "probability_raw_good_probability": good_probability,
                "bad_prediction_confidence": confidence,
            }
        )

    return pd.DataFrame(rows).sort_values("sample_idx").reset_index(drop=True)


def load_x_test(x_test_path: Path) -> pd.DataFrame:
    if not x_test_path.exists():
        raise FileNotFoundError(f"X_test file not found: {x_test_path}")

    x_test = pd.read_csv(x_test_path).copy()
    x_test.insert(0, "sample_idx", range(len(x_test)))
    return x_test


def main():
    args = parse_args()

    shap_dir = resolve_path(args.shap_dir)
    x_test_path = resolve_path(args.x_test)
    output_path = resolve_path(args.output)

    confidence_df = load_shap_jsons(shap_dir)

    warning_df = load_confidence_csv(
        resolve_path(args.confidence_csv)
    )

    x_test = load_x_test(x_test_path)

    merged = confidence_df.merge(
        x_test,
        on="sample_idx",
        how="left",
        validate="one_to_one",
    )
    
    merged = merged.merge(
        warning_df,
        on="sample_idx",
        how="left",
    )

    if merged[FEATURE_COLUMNS].isna().any().any():
        missing = [
            col for col in FEATURE_COLUMNS
            if col not in merged.columns or merged[col].isna().any()
        ]
        print("Warning: some requested feature columns are missing or contain NaN:")
        print(missing)

    sigma = merged["probability_raw_bad_probability"].std(ddof=0)
    margin = sigma * args.sigma_multiplier

    lower_bound = args.decision_boundary - margin
    upper_bound = args.decision_boundary + margin

    merged["decision_boundary_distance"] = (
        merged["probability_raw_bad_probability"] - args.decision_boundary
    )

    merged["decision_boundary_abs_distance"] = (
        merged["decision_boundary_distance"].abs()
    )

    within_mask = merged["probability_raw_bad_probability"].between(
        lower_bound,
        upper_bound,
        inclusive="both",
    )

    merged["distance"] = "far"
    merged.loc[within_mask, "distance"] = "near"

    output_cols = [
        "sample_idx",
        "distance",
        "warning_type",
        "decision_boundary_abs_distance",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_label",
        "is_correct",
        "probability_raw_bad_probability",
        "probability_raw_good_probability",
        "bad_prediction_confidence",
    ]

    output_cols += [f"feature_{col}" for col in FEATURE_COLUMNS]

    rename_map = {
        col: f"feature_{col}"
        for col in FEATURE_COLUMNS
    }

    final_df = merged.rename(columns=rename_map)

    final_df = final_df[output_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("=== Saved decision boundary feature file ===")
    print(f"Input SHAP dir: {shap_dir}")
    print(f"X_test: {x_test_path}")
    print(f"Output: {output_path}")
    print(f"Total samples: {len(final_df)}")
    print(f"Decision boundary: {args.decision_boundary}")
    print(f"Sigma: {sigma:.6f}")
    print(f"Near range: {lower_bound:.6f} ~ {upper_bound:.6f}")
    print(final_df["distance"].value_counts())


if __name__ == "__main__":
    main()