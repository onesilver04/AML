import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


DROP_COLUMNS = [
    "foreign_worker_no",
    "foreign_worker_yes",
    "num_dependents",
    "own_telephone_none",
    "own_telephone_yes",
    "personal_status_female div/dep/mar",
    "personal_status_male div/sep",
    "personal_status_male mar/wid",
    "personal_status_male single",
    "other_parties_co applicant",
    "other_parties_guarantor",
    "other_parties_none",
    "property_magnitude_car",
    "property_magnitude_life insurance",
    "property_magnitude_no known property",
    "property_magnitude_real estate",
    "other_payment_plans_bank",
    "other_payment_plans_none",
    "other_payment_plans_stores",
    "residence_since",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF and save global SHAP importance for raw and grouped features."
    )
    parser.add_argument("--data-path", default="german21_ohe.csv")
    parser.add_argument(
        "--output-dir",
        default="SHAP/Global SHAP",
        help="Directory where global SHAP outputs will be saved.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.80,
        help="Ratio of features kept after permutation importance filtering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to include in the JSON summary.",
    )
    return parser.parse_args()


def load_target(df: pd.DataFrame) -> pd.Series:
    if "class" not in df.columns:
        raise ValueError("타깃 컬럼 'class'를 찾지 못했습니다.")

    if df["class"].dtype == bool:
        return (~df["class"]).astype(int)

    vals = set(pd.Series(df["class"]).dropna().unique())
    if vals.issubset({1, 2}):
        return (df["class"] == 2).astype(int)
    if vals.issubset({0, 1}):
        return df["class"].astype(int)
    raise ValueError(f"예상치 못한 class 값들: {vals}")


def ohe_prefix(name: str) -> str:
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def evaluate_model(y_true: pd.Series, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_score": float(f1_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, proba)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def group_shap_by_prefix(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    prefix_to_indices = {}
    for idx, feature in enumerate(feature_names):
        prefix_to_indices.setdefault(ohe_prefix(feature), []).append(idx)

    grouped_rows = []
    for prefix, indices in prefix_to_indices.items():
        grouped_values = shap_values[:, indices].sum(axis=1)
        grouped_rows.append(
            {
                "prefix": prefix,
                "mean_abs_shap": float(np.mean(np.abs(grouped_values))),
                "mean_shap": float(np.mean(grouped_values)),
                "num_features": len(indices),
            }
        )

    return (
        pd.DataFrame(grouped_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = load_target(df)
    X = df.drop(columns=["class"], errors="ignore")

    print("전체 분포(원본):")
    print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_full = RandomForestClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    rf_full.fit(X_train, y_train)

    full_metrics = evaluate_model(y_test, rf_full.predict_proba(X_test)[:, 1])
    print("\n=== Baseline RF (ALL features) ===")
    print(json.dumps(full_metrics, indent=2))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    rf_tmp = RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)
    rf_tmp.fit(X_tr, y_tr)

    perm = permutation_importance(
        rf_tmp,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    importances = pd.Series(perm.importances_mean, index=X_val.columns).sort_values(
        ascending=False
    )
    num_keep = max(1, int(len(importances) * args.keep_ratio))
    selected_features = importances.index[:num_keep].tolist()

    print(f"\n선택된 피처 개수: {len(selected_features)}")
    print(importances.head(15))

    rf_selected = RandomForestClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    rf_selected.fit(X_train[selected_features], y_train)

    selected_proba = rf_selected.predict_proba(X_test[selected_features])[:, 1]
    selected_metrics = evaluate_model(y_test, selected_proba)
    print("\n=== RF (Selected features) ===")
    print(json.dumps(selected_metrics, indent=2))

    explainer = shap.TreeExplainer(rf_selected)
    shap_exp = explainer(X_test[selected_features])
    if len(shap_exp.values.shape) == 3:
        shap_exp = shap_exp[:, :, 1]

    shap_values = shap_exp.values
    raw_importance_df = (
        pd.DataFrame(
            {
                "feature": selected_features,
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "mean_shap": np.mean(shap_values, axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    prefix_importance_df = group_shap_by_prefix(shap_values, selected_features)

    raw_csv_path = output_dir / "global_shap_raw_feature_importance.csv"
    prefix_csv_path = output_dir / "global_shap_prefix_importance.csv"
    summary_json_path = output_dir / "global_shap_summary.json"

    raw_importance_df.to_csv(raw_csv_path, index=False)
    prefix_importance_df.to_csv(prefix_csv_path, index=False)

    summary = {
        "data_path": args.data_path,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "keep_ratio": args.keep_ratio,
        "selected_feature_count": len(selected_features),
        "selected_metrics": selected_metrics,
        "top_raw_features": raw_importance_df.head(args.top_k).to_dict(orient="records"),
        "top_prefix_groups": prefix_importance_df.head(args.top_k).to_dict(orient="records"),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTop raw global SHAP:")
    print(raw_importance_df.head(args.top_k).to_string(index=False))

    print("\nTop prefix global SHAP:")
    print(prefix_importance_df.head(args.top_k).to_string(index=False))

    print(f"\nSaved: {raw_csv_path}")
    print(f"Saved: {prefix_csv_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
