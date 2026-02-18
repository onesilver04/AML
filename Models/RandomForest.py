import os
import json
import time
import traceback
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =========================
# Matplotlib (save only)
# =========================
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
DATA_PATH = "german21_ohe.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
N_ITER = 25
SEARCH_N_JOBS = 1
CVPRED_N_JOBS = 1

# 실험용: 제거할 그룹(prefix) 지정
# - 예: ["foreign"], ["foreign","existing"], [] 등
DROP_GROUPS = ["foreign", "existing", "num", "own", "personal", "residence", "other", "property"]

# (옵션) 이 컬럼은 따로 제거 실험을 하고 있었다고 해서 기본 제거 유지
DROP_INSTALLMENT_COMMITMENT = True


# =========================
# Helpers
# =========================
def parse_target_class(df: pd.DataFrame, target_col: str = "class") -> pd.Series:
    """
    Robust target parsing:
    - bool True/False -> int (False=0, True=1)
    - numeric (1/2) -> (==2) as bad=1
    - string good/bad -> {good:0, bad:1}
    """
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 `{target_col}`를 찾지 못했습니다.")

    raw_y = df[target_col]

    # bool (TRUE/FALSE)
    if pd.api.types.is_bool_dtype(raw_y):
        return raw_y.astype(int)  # False=0, True=1

    # numeric (possibly 1=good, 2=bad)
    if pd.api.types.is_numeric_dtype(raw_y):
        uniq = pd.Series(raw_y.dropna().unique()).sort_values().tolist()
        if set(uniq).issubset({0, 1}):
            return raw_y.astype(int)
        return (raw_y == 2).astype(int)

    # string (good/bad)
    y = (
        raw_y.astype(str)
        .str.strip()
        .str.lower()
        .map({"good": 0, "bad": 1, "false": 0, "true": 1})
    )
    if y.isna().any():
        bad_vals = raw_y[y.isna()].astype(str).value_counts().head(10)
        raise ValueError(f"`class`에 예상 밖 값이 있습니다. 상위 예시:\n{bad_vals}")
    return y.astype(int)


def make_ohe_encoder():
    """
    sklearn 버전 호환:
    - 신규: sparse_output=False
    - 구버전: sparse=False
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def coerce_bool_features_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """
    SimpleImputer가 bool dtype을 지원하지 않는 경우가 있어,
    bool feature는 0/1 int로 변환.
    """
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(np.int8)
    return X


def build_preprocess(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    - OHE가 이미 되어 있으면: 대부분 numeric/bool -> numeric 파이프라인만
    - object/category가 있으면: categorical 파이프라인 + OHE 적용
    """
    X = coerce_bool_features_to_int(X)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    if len(categorical_cols) == 0:
        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
            ],
            remainder="drop",
        )
        return preprocess, numeric_cols, categorical_cols

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe_encoder()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess, numeric_cols, categorical_cols


def drop_prefix_groups(X: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    그룹(prefix) 리스트를 받아서, prefix_ 로 시작하는 컬럼들을 통째로 제거.
    반환: (X_new, removed_counts)
    """
    X_new = X.copy()
    removed_counts = {}

    for g in groups:
        cols = [c for c in X_new.columns if c == g or c.startswith(g + "_")]
        removed_counts[g] = len(cols)
        if cols:
            X_new = X_new.drop(columns=cols, errors="ignore")

    return X_new, removed_counts


def train_eval_once(X: pd.DataFrame, y: pd.Series, tag: str) -> dict:
    """
    한 번 학습/튜닝/평가하고 결과 dict 반환.
    """
    X = coerce_bool_features_to_int(X)

    preprocess, numeric_cols, categorical_cols = build_preprocess(X)

    print(f"\n[{tag}] 컬럼 타입 요약:")
    print(f"- 수치형: {len(numeric_cols)}개")
    print(f"- 범주형: {len(categorical_cols)}개")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", base_model),
        ]
    )

    param_distributions = {
        "model__n_estimators": [300, 500, 800, 1200],
        "model__max_depth": [None, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40],
        "model__min_samples_split": [2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50],
        "model__min_samples_leaf": [1, 2, 3, 4, 5, 8, 10, 15, 20],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9, 1.0],
        "model__bootstrap": [True, False],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="accuracy",
        n_jobs=SEARCH_N_JOBS,
        cv=cv,
        refit=True,
        verbose=1 if tag == "BASELINE" else 0,  # baseline만 로그 조금
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # threshold 튜닝(OOF)
    oof_proba = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=CVPRED_N_JOBS,
    )[:, 1]

    thresholds = np.linspace(0.05, 0.95, 181)
    oof_accs = np.array([accuracy_score(y_train, (oof_proba >= t).astype(int)) for t in thresholds])
    best_thr = float(thresholds[int(oof_accs.argmax())])
    best_oof_acc = float(oof_accs.max())

    # test
    best_model.fit(X_train, y_train)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_thr = (y_proba >= best_thr).astype(int)

    acc_thr = accuracy_score(y_test, y_pred_thr)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thr).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    result = {
        "tag": tag,
        "best_cv_acc": float(search.best_score_),
        "best_params": search.best_params_,
        "best_thr": best_thr,
        "best_oof_acc": best_oof_acc,
        "test_acc_thr": float(acc_thr),
        "test_auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "feature_cols": X.columns.tolist(),
    }
    return result


def print_feature_importance(best_model: Pipeline, feature_cols: list[str], top_n: int = 30) -> None:
    """
    OHE 데이터(모두 numeric) 기준으로:
    - feature importance top_n 출력
    - prefix 그룹 importance 출력
    """
    rf = best_model.named_steps["model"]
    importances = rf.feature_importances_
    feature_names = feature_cols

    # individual
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n=== Feature importance (top {top_n}) ===")
    for name, imp in feat_imp[:top_n]:
        print(f"{name:40s} : {imp:.4f}")

    # grouped by prefix
    from collections import defaultdict

    group_imp = defaultdict(float)
    for name, imp in zip(feature_names, importances):
        if "_" in name:
            group = name.split("_", 1)[0]
        else:
            group = name
        group_imp[group] += float(imp)

    group_imp_sorted = sorted(group_imp.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Grouped feature importance ===")
    for g, imp in group_imp_sorted:
        print(f"{g:25s} : {imp:.4f}")


def save_roc_curve(y_test: pd.Series, y_proba: np.ndarray, out_path: str, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# Main
# =========================
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    
    # 1) target
    y = parse_target_class(df, target_col="class")

    print("전체 분포(원본):")
    print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))

    if y.nunique() < 2:
        raise ValueError(
            f"타깃 y가 한 클래스만 포함합니다. (unique={y.unique().tolist()})\n"
            f"`class`가 TRUE/FALSE인 경우, TRUE가 1로 가는지 확인하세요."
        )

    # 2) X
    X = df.drop(columns=["class"], errors="ignore")

    # (optional) drop installment_commitment
    if DROP_INSTALLMENT_COMMITMENT and "installment_commitment" in X.columns:
        print("\n`installment_commitment` 컬럼을 제거하고 모델을 학습합니다.")
        X = X.drop(columns=["installment_commitment"])
    else:
        print("\n`installment_commitment` 컬럼이 존재하지 않습니다(혹은 제거 옵션 off). 그대로 진행합니다.")

    # bool -> int
    X = coerce_bool_features_to_int(X)

    # baseline run
    print("\n============================")
    print("RUN 1) BASELINE (no group drop)")
    print("============================")
    res_base = train_eval_once(X, y, tag="BASELINE")

    print("\n=== RandomizedSearchCV 결과(Train CV) ===")
    print(f"- Best CV Accuracy: {res_base['best_cv_acc']:.4f}")
    print(f"- Best Params: {res_base['best_params']}")

    print("\n=== Threshold 튜닝(Train OOF) ===")
    print(f"- Best threshold (accuracy): {res_base['best_thr']:.3f}")
    print(f"- OOF accuracy @ best thr : {res_base['best_oof_acc']:.4f}")

    print("\n=== Metrics (RandomForest) on ORIGINAL test distribution ===")
    print(f"Accuracy(thr tuned) : {res_base['test_acc_thr']:.4f} (thr={res_base['best_thr']:.3f})")
    print(f"AUC                : {res_base['test_auc']:.4f}")
    print(f"Sensitivity        : {res_base['sensitivity']:.4f} (TPR, Recall of bad=1)")
    print(f"Specificity        : {res_base['specificity']:.4f} (TNR, Recall of good=0)")
    print(f"Confusion          : TN={res_base['tn']}, FP={res_base['fp']}, FN={res_base['fn']}, TP={res_base['tp']}")

    print_feature_importance(res_base["best_model"], res_base["feature_cols"], top_n=30)

    save_roc_curve(
        res_base["y_test"],
        res_base["y_proba"],
        out_path="roc_curve_rf_baseline.png",
        title="ROC Curve (RandomForest, baseline)",
    )

    # drop group(s) run
    if DROP_GROUPS:
        X_drop, removed_counts = drop_prefix_groups(X, DROP_GROUPS)
        print("\n============================")
        print(f"RUN 2) DROP groups = {DROP_GROUPS}")
        print("============================")
        print("제거된 컬럼 개수:", removed_counts)
        print("feature 수 변화:", X.shape[1], "->", X_drop.shape[1])

        res_drop = train_eval_once(X_drop, y, tag=f"DROP_{'_'.join(DROP_GROUPS)}")

        print("\n=== Metrics (after drop) ===")
        print(f"Accuracy(thr tuned) : {res_drop['test_acc_thr']:.4f} (thr={res_drop['best_thr']:.3f})")
        print(f"AUC                : {res_drop['test_auc']:.4f}")
        print(f"Sensitivity        : {res_drop['sensitivity']:.4f}")
        print(f"Specificity        : {res_drop['specificity']:.4f}")
        print(f"Confusion          : TN={res_drop['tn']}, FP={res_drop['fp']}, FN={res_drop['fn']}, TP={res_drop['tp']}")

        print("\n=== Compare (DROP - BASELINE) ===")
        print(f"ΔACC = {res_drop['test_acc_thr'] - res_base['test_acc_thr']:+.4f}")
        print(f"ΔAUC = {res_drop['test_auc'] - res_base['test_auc']:+.4f}")

        save_roc_curve(
            res_drop["y_test"],
            res_drop["y_proba"],
            out_path="roc_curve_rf_drop.png",
            title=f"ROC Curve (RandomForest, drop={','.join(DROP_GROUPS)})",
        )

        # 저장(드랍 실험 모델)
        joblib.dump(res_drop["best_model"], "rf_model_german21_drop.joblib")
        res_drop["X_test"].to_csv("X_test_drop.csv", index=False)
        res_drop["y_test"].to_csv("y_test_drop.csv", index=False)

        print("\nSaved (drop run):")
        print("- roc_curve_rf_drop.png")
        print("- rf_model_german21_drop.joblib")
        print("- X_test_drop.csv")
        print("- y_test_drop.csv")

    # 저장(베이스라인 모델)
    joblib.dump(res_base["best_model"], "rf_model_german21_baseline.joblib")
    res_base["X_test"].to_csv("X_test_baseline.csv", index=False)
    res_base["y_test"].to_csv("y_test_baseline.csv", index=False)

    print("\nSaved (baseline run):")
    print("- roc_curve_rf_baseline.png")
    print("- rf_model_german21_baseline.joblib")
    print("- X_test_baseline.csv")
    print("- y_test_baseline.csv")


if __name__ == "__main__":
    main()
