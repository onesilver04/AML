import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = "german21_ohe.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

N_ITER = 300          # 더 올릴수록 탐색 잘 됨 (시간 되면 500도 OK)
SEARCH_N_JOBS = 1
CVPRED_N_JOBS = 1

DROP_GROUPS = ["foreign", "existing", "num", "own", "personal", "residence", "other", "property"]
DROP_INSTALLMENT_COMMITMENT = True


def parse_target_class(df: pd.DataFrame, target_col: str = "class") -> pd.Series:
    raw_y = df[target_col]
    if pd.api.types.is_bool_dtype(raw_y):
        return raw_y.astype(int)
    if pd.api.types.is_numeric_dtype(raw_y):
        uniq = pd.Series(raw_y.dropna().unique()).sort_values().tolist()
        if set(uniq).issubset({0, 1}):
            return raw_y.astype(int)
        return (raw_y == 2).astype(int)
    y = (
        raw_y.astype(str).str.strip().str.lower().map({"good": 0, "bad": 1, "false": 0, "true": 1})
    )
    if y.isna().any():
        raise ValueError("class 컬럼에 예상 밖 값이 있습니다.")
    return y.astype(int)


def coerce_bool_features_to_int(X: pd.DataFrame) -> pd.DataFrame:
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(np.int8)
    return X


def drop_prefix_groups(X: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    X_new = X.copy()
    for g in groups:
        cols = [c for c in X_new.columns if c == g or c.startswith(g + "_")]
        X_new = X_new.drop(columns=cols, errors="ignore")
    return X_new


def tuned_threshold_accuracy(estimator, X_val, y_val) -> float:
    """
    RandomizedSearchCV scoring용 callable.
    fold의 validation에서 threshold를 스캔해 accuracy 최대값을 반환.
    """
    proba = estimator.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    best = 0.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best:
            best = acc
    return float(best)


def main():
    df = pd.read_csv(DATA_PATH)
    y = parse_target_class(df, "class")
    X = df.drop(columns=["class"], errors="ignore")
    X = coerce_bool_features_to_int(X)

    if DROP_INSTALLMENT_COMMITMENT and "installment_commitment" in X.columns:
        X = X.drop(columns=["installment_commitment"])

    if DROP_GROUPS:
        X = drop_prefix_groups(X, DROP_GROUPS)

    print("전체 분포(원본):")
    print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))
    print("X shape:", X.shape)

    # ✅ 여기 split은 seed 고정이라 매번 동일합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), X.columns.tolist()),
        ],
        remainder="drop",
    )

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", base_model),
        ]
    )

    # ✅ “현재 best 근처로 집중”한 탐색 공간 (정확도 끌어올릴 때 유리)
    param_distributions = {
        "model__n_estimators": [800, 1200, 2000, 3000, 4000],
        "model__max_depth": [4, 5, 6, 7, 8, 9, 10, None],
        "model__min_samples_split": [10, 15, 20, 25, 30, 40],
        "model__min_samples_leaf": [2, 3, 4, 5, 6, 8, 10],
        "model__max_features": [0.3, 0.5, 0.7, 0.9, 1.0, "sqrt", "log2"],
        "model__bootstrap": [True],
        "model__class_weight": [None],  # accuracy 목표면 일단 None 고정 권장
        "model__criterion": ["gini", "entropy", "log_loss"],
    }

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring=tuned_threshold_accuracy,   # ✅ 핵심 변경점
        n_jobs=SEARCH_N_JOBS,
        cv=cv,
        refit=True,
        verbose=1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\n=== Search 결과 (CV tuned-threshold accuracy) ===")
    print(f"- Best CV score: {search.best_score_:.4f}")
    print(f"- Best Params: {search.best_params_}")

    # ✅ train에서 OOF로 threshold 튜닝 (최종 threshold)
    oof_proba = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=CVPRED_N_JOBS
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    oof_accs = np.array([accuracy_score(y_train, (oof_proba >= t).astype(int)) for t in thresholds])
    best_thr = float(thresholds[int(oof_accs.argmax())])

    # test 평가
    best_model.fit(X_train, y_train)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n=== TEST (fixed 80/20 split) ===")
    print(f"Accuracy(tuned thr): {acc:.4f} (thr={best_thr:.3f})")
    print(f"AUC              : {auc:.4f}")
    print(f"Confusion        : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    joblib.dump(best_model, "rf_best_tuned.joblib")
    print("\nSaved: rf_best_tuned.joblib")


if __name__ == "__main__":
    main()
