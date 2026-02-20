# uci 데이터 + SMOTE + RF + GridSearchCV
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# =========================
# 1) Load
# =========================
data = pd.read_csv("german21_ohe.csv")
print("shape:", data.shape)

# =========================
# 2) Drop columns (네가 하던 방식 유지)
# =========================
drop_cols = [
    "foreign_worker_no", "foreign_worker_yes",
    "other_parties_co applicant", "other_parties_guarantor", "other_parties_none",
    "num_dependents"
]
df = data.drop(columns=[c for c in drop_cols if c in data.columns])
print("after drop shape:", df.shape)

# =========================
# 3) Target split
# =========================
target = "class"
if target not in df.columns:
    raise ValueError(f"target column '{target}' not found. Available cols: {df.columns.tolist()[:30]} ...")

X = df.drop(columns=[target])
y = df[target]

# 타겟이 0/1인지 확인 (혹시 'good'/'bad'면 매핑)
if y.dtype == "object":
    # 보통 German credit에서는 good/bad가 있을 수 있어서 안전장치
    y = y.map({"good": 1, "bad": 0}).astype(int)

print("X shape:", X.shape, "y dist:", y.value_counts(dropna=False).to_dict())

# =========================
# 4) Train/Test split (테스트는 마지막 평가용)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 5) Pipeline: Scaling -> SMOTE -> RF
#    (중요: GridSearchCV가 CV fold별로 scaler/SMOTE를 train fold에서만 fit/적용)
# =========================
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),             # RF는 필수 아니지만, 네가 원한대로 포함
    ("smote", SMOTE(random_state=42)),        # 불균형 보정
    ("rf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

# =========================
# 6) GridSearchCV
#    - 논문처럼 5-fold
#    - scoring은 roc_auc 추천(신용위험), F1도 같이 보고 refit은 roc_auc로
# =========================
param_grid = {
    "rf__n_estimators": [300, 600, 1000],
    "rf__max_depth": [None, 6, 10, 14, 20],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", None],
    "rf__bootstrap": [True, False],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
    refit="roc_auc",          # 최종 best 모델 선택 기준
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid.fit(X_train, y_train)

print("\n===== BEST (refit = roc_auc) =====")
print("Best CV ROC-AUC:", grid.best_score_)
print("Best Params:", grid.best_params_)

best_model = grid.best_estimator_

# =========================
# 7) Test evaluation (hold-out test)
# =========================
# 예측 확률/라벨
proba = best_model.predict_proba(X_test)[:, 1]
pred = best_model.predict(X_test)

print("\n===== TEST RESULTS =====")
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("F1:", f1_score(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred, digits=4))

# =========================
# 8) (선택) Baseline 비교: SMOTE 없이 RF만 grid로 돌려보고 싶으면
#    - 아래 블록을 켜서 비교하면 "SMOTE가 진짜 도움이 됐는지" 결론이 깔끔해짐
# =========================
"""
pipe_nosmote = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_nosmote = {k.replace("rf__", "rf__"): v for k, v in param_grid.items()}  # 동일 grid 사용

grid_nosmote = GridSearchCV(
    estimator=pipe_nosmote,
    param_grid=param_grid,   # rf__ 파라미터는 그대로 사용 가능
    cv=cv,
    scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
    refit="roc_auc",
    n_jobs=-1,
    verbose=2
)
grid_nosmote.fit(X_train, y_train)

best_nosmote = grid_nosmote.best_estimator_
proba_ns = best_nosmote.predict_proba(X_test)[:, 1]
pred_ns = best_nosmote.predict(X_test)

print("\n===== BASELINE (NO SMOTE) TEST =====")
print("ROC-AUC:", roc_auc_score(y_test, proba_ns))
print("F1:", f1_score(y_test, pred_ns))
print("Accuracy:", accuracy_score(y_test, pred_ns))
"""