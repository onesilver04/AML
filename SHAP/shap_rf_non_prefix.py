# prefix 없는 원본 RF 학습/평가 및 글로벌 SHAP 저장

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import shap

from make_llm_input_with_definitions import get_feature_definition


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


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

RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 800, 1200],
    "max_depth": [None, 6, 8, 10, 14, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None, 0.5, 0.8],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate RF, save global SHAP importance, and export random local SHAP samples."
    )
    parser.add_argument("--data-path", default="german21_ohe.csv")
    parser.add_argument(
        "--random-samples",
        type=int,
        default=100,
        help="Randomly select this many local SHAP samples from X_test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random-samples.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated sample indices. Overrides --random-samples.",
    )
    parser.add_argument(
        "--sample-index-file",
        type=str,
        default=None,
        help="File containing sample indices separated by commas or newlines.",
    )
    parser.add_argument(
        "--output-dir",
        default="SHAP/100 Local Shap",
        help="Directory where shap_tuples_non_prefix_{sample_idx}.json files are saved.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip global SHAP summary plots.",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=32,
        help="Number of random RF hyperparameter candidates evaluated with CV.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds used for hyperparameter search and threshold tuning.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.20,
        help="Minimum threshold value searched on out-of-fold train probabilities.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
        help="Maximum threshold value searched on out-of-fold train probabilities.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step size.",
    )
    return parser.parse_args()


def parse_index_text(text: str):
    return [
        int(part.strip())
        for part in text.replace("\n", ",").split(",")
        if part.strip()
    ]


def select_sample_indices(args, n_samples: int):
    if args.sample_indices:
        sample_indices = parse_index_text(args.sample_indices)
    elif args.sample_index_file:
        with open(args.sample_index_file, "r", encoding="utf-8") as f:
            sample_indices = parse_index_text(f.read())
    else:
        if args.random_samples < 1:
            raise ValueError("--random-samples must be at least 1.")
        if args.random_samples > n_samples:
            raise ValueError(
                f"--random-samples={args.random_samples} exceeds available "
                f"test samples ({n_samples})."
            )
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(
            n_samples,
            size=args.random_samples,
            replace=False,
        ).tolist()

    invalid_sample_indices = [
        idx for idx in sample_indices if idx < 0 or idx >= n_samples
    ]
    if invalid_sample_indices:
        raise IndexError(
            f"유효하지 않은 sample index: {invalid_sample_indices}. "
            f"가능한 범위는 0 ~ {n_samples - 1} 입니다."
        )
    return sample_indices


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def evaluate_split_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion": (tn, fp, fn, tp),
        "threshold": threshold,
    }


def print_metrics(title: str, train_metrics: dict, test_metrics: dict):
    print(f"\n=== {title} ===")
    print(
        f"Train Threshold: {train_metrics.get('threshold', 0.5):.2f} | "
        f"Test Threshold : {test_metrics.get('threshold', 0.5):.2f}"
    )
    print(
        f"Train Accuracy : {train_metrics['accuracy']:.4f} | "
        f"Test Accuracy : {test_metrics['accuracy']:.4f}"
    )
    print(
        f"Train F1       : {train_metrics['f1']:.4f} | "
        f"Test F1       : {test_metrics['f1']:.4f}"
    )
    print(
        f"Train AUC      : {train_metrics['auc']:.4f} | "
        f"Test AUC      : {test_metrics['auc']:.4f}"
    )
    print(
        f"Train Sens.    : {train_metrics['sensitivity']:.4f} | "
        f"Test Sens.    : {test_metrics['sensitivity']:.4f}"
    )
    print(
        f"Train Spec.    : {train_metrics['specificity']:.4f} | "
        f"Test Spec.    : {test_metrics['specificity']:.4f}"
    )
    print(
        "Train Confusion: "
        f"TN={train_metrics['confusion'][0]}, FP={train_metrics['confusion'][1]}, "
        f"FN={train_metrics['confusion'][2]}, TP={train_metrics['confusion'][3]}"
    )
    print(
        "Test Confusion : "
        f"TN={test_metrics['confusion'][0]}, FP={test_metrics['confusion'][1]}, "
        f"FN={test_metrics['confusion'][2]}, TP={test_metrics['confusion'][3]}"
    )


def print_tuning_result(tuned: dict):
    print("\n=== RF Tuning Result (ALL features) ===")
    print(f"Best params        : {tuned['params']}")
    print(f"Best CV AUC        : {tuned['cv_auc']:.4f}")
    print(f"Best threshold     : {tuned['threshold']:.2f}")
    print(
        f"OOF Train AUC/F1   : {tuned['oof_metrics']['auc']:.4f} / "
        f"{tuned['oof_metrics']['f1']:.4f}"
    )
    print(
        f"OOF Train Accuracy : {tuned['oof_metrics']['accuracy']:.4f}"
    )


def to_risk_label(value: int) -> str:
    return "BAD CREDIT RISK" if int(value) == 1 else "GOOD CREDIT RISK"


def build_threshold_grid(args) -> np.ndarray:
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be > 0.")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be smaller than --threshold-max.")

    return np.round(
        np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step),
        4,
    )


def make_rf(random_state: int = 42, n_jobs: int = -1, **params):
    return RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )


def tune_rf(X_train, y_train, args):
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=42,
    )
    threshold_grid = build_threshold_grid(args)

    search = RandomizedSearchCV(
        estimator=make_rf(n_jobs=1),
        param_distributions=RF_PARAM_DISTRIBUTIONS,
        n_iter=args.search_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    oof_proba = cross_val_predict(
        make_rf(n_jobs=1, **best_params),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    best_threshold_metrics = None
    for threshold in threshold_grid:
        metrics = evaluate_split_at_threshold(y_train, oof_proba, threshold)
        score = (
            metrics["accuracy"],
            metrics["f1"],
            metrics["specificity"],
            -abs(threshold - 0.5),
        )
        if best_threshold_metrics is None or score > best_threshold_metrics["score"]:
            best_threshold_metrics = {
                "score": score,
                "threshold": threshold,
                "metrics": metrics,
            }

    return {
        "params": best_params,
        "cv_auc": search.best_score_,
        "threshold": best_threshold_metrics["threshold"],
        "oof_metrics": best_threshold_metrics["metrics"],
    }


def ohe_prefix(name: str) -> str:
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def group_shap_values_by_prefix(sv: np.ndarray, feature_names: list[str]):
    prefix_codes, unique_prefixes = pd.factorize(
        np.asarray([ohe_prefix(name) for name in feature_names], dtype=object),
        sort=False,
    )
    grouped_sv = np.zeros((sv.shape[0], len(unique_prefixes)), dtype=sv.dtype)
    for feature_idx, prefix_idx in enumerate(prefix_codes):
        grouped_sv[:, prefix_idx] += sv[:, feature_idx]
    return grouped_sv, unique_prefixes


def build_raw_shap_records(
    sv: np.ndarray,
    feature_names: list[str],
    sample_idx: int,
    top_k: int = 3,
):
    sample_vals = sv[sample_idx]
    top_indices = np.argsort(np.abs(sample_vals))[::-1][:top_k]

    records = []
    for feature_idx in top_indices:
        shap_value = float(sample_vals[feature_idx])
        records.append(
            {
                "feature": feature_names[feature_idx],
                "definition": get_feature_definition(feature_names[feature_idx]),
                "shap_value": shap_value,
                "abs_shap": abs(shap_value),
                "direction": "increase_risk" if shap_value >= 0 else "decrease_risk",
            }
        )
    return records


def save_shap_tuples_json(
    records,
    sample_idx,
    true_label,
    prediction_label,
    predict_proba=None,
    save_path="SHAP/shap_tuples_non_prefix.json",
):
    data = {
        "sample_idx": sample_idx,
        "true_label": true_label,
        "prediction": {
            "label": prediction_label,
            "probability": predict_proba,
        },
        "tuples": records,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {save_path}")


def main():
    args = parse_args()
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = load_target(df)
    X = df.drop(columns=["class"], errors="ignore")

    print("전체 분포(원본):")
    print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    sample_indices = select_sample_indices(args, len(X_test))
    print("\nSelected sample indices:")
    print(sample_indices)

    print("\nTrain 분포:")
    print(y_train.value_counts().rename({0: "good(0)", 1: "bad(1)"}))
    print("\nTest 분포(현실 분포 유지):")
    print(y_test.value_counts().rename({0: "good(0)", 1: "bad(1)"}))

    tuned = tune_rf(X_train, y_train, args)
    print_tuning_result(tuned)

    rf = make_rf(**tuned["params"])
    rf.fit(X_train, y_train)

    proba_train = rf.predict_proba(X_train)[:, 1]
    proba_test = rf.predict_proba(X_test)[:, 1]
    train_metrics = evaluate_split_at_threshold(y_train, proba_train, tuned["threshold"])
    test_metrics = evaluate_split_at_threshold(y_test, proba_test, tuned["threshold"])
    print_metrics("RF (ALL features)", train_metrics, test_metrics)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )
    rf_tmp = make_rf(**tuned["params"])
    rf_tmp.fit(X_tr, y_tr)

    perm = permutation_importance(
        rf_tmp,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importances = pd.Series(
        perm.importances_mean,
        index=X_val.columns,
    ).sort_values(ascending=False)

    print("\nTop 15 importances:")
    print(importances.head(15))

    explainer = shap.TreeExplainer(rf)
    shap_exp = explainer(X_test)

    if len(shap_exp.values.shape) == 3:
        shap_exp = shap_exp[:, :, 1]

    if not args.skip_plots:
        shap.plots.beeswarm(shap_exp, max_display=20)
        plt.show()

        shap.plots.bar(shap_exp, max_display=20)
        plt.show()

    sv = shap_exp.values
    feature_names = X_train.columns.tolist()

    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 20 SHAP feature importance (raw features):")
    print(imp_df.head(20).to_string(index=False))

    raw_output_path = PROJECT_ROOT / "SHAP/selected_shap_feature_importance.csv"
    imp_df.to_csv(raw_output_path, index=False)
    print(f"\nSaved: {raw_output_path}")

    grouped_sv, unique_prefixes = group_shap_values_by_prefix(sv, feature_names)
    grouped_mean_abs = np.mean(np.abs(grouped_sv), axis=0)

    group_imp_df = (
        pd.DataFrame({"prefix": unique_prefixes, "mean_abs_shap": grouped_mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print("\nprefix 개수:", len(unique_prefixes))
    print("\nTop 20 SHAP importance (Grouped by prefix):")
    print(group_imp_df.head(20).to_string(index=False))

    prefix_output_path = PROJECT_ROOT / "SHAP/selected_shap_prefix_importance.csv"
    group_imp_df.to_csv(prefix_output_path, index=False)
    print(f"\nSaved: {prefix_output_path}")

    top_k = 3
    for sample_idx in sample_indices:
        sample_records = build_raw_shap_records(
            sv=sv,
            feature_names=feature_names,
            sample_idx=sample_idx,
            top_k=top_k,
        )

        true_label = to_risk_label(int(y_test.iloc[sample_idx]))
        prediction_label = (
            "BAD CREDIT RISK"
            if proba_test[sample_idx] >= tuned["threshold"]
            else "GOOD CREDIT RISK"
        )
        prediction_proba = float(proba_test[sample_idx])

        save_shap_tuples_json(
            sample_records,
            sample_idx,
            true_label=true_label,
            prediction_label=prediction_label,
            predict_proba=prediction_proba,
            save_path=output_dir / f"shap_tuples_non_prefix_{sample_idx}.json",
        )


if __name__ == "__main__":
    main()
