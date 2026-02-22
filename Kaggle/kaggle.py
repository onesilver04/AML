import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

import matplotlib.pyplot as plt


# =========================
# 0) 데이터 로드
# =========================
DATA_PATH = "german21_ohe.csv"
df = pd.read_csv(DATA_PATH)

# 외국인 컬럼 제거: 네가 이미 제거했다고 했지만, 파일에 남아있으면 제거
df = df.drop(columns=["foreign_worker_no", "foreign_worker_yes", "num_dependents", "own_telephone_none", "own_telephone_yes", "personal_status_female div/dep/mar","personal_status_male div/sep","personal_status_male mar/wid","personal_status_male single","other_parties_co applicant","other_parties_guarantor","other_parties_none","property_magnitude_car","property_magnitude_life insurance","property_magnitude_no known property","property_magnitude_real estate","other_payment_plans_bank","other_payment_plans_none","other_payment_plans_stores", "residence_since"], errors="ignore")

# =========================
# 타깃 설정 (bad=1, good=0) - 이 데이터에 맞춤
# =========================
if "class" not in df.columns:
    raise ValueError("타깃 컬럼 'class'를 찾지 못했습니다.")

# german21_ohe.csv의 class는 True/False (True가 700개로 good에 해당)
# bad=1, good=0 으로 만들기
if df["class"].dtype == bool:
    y = (~df["class"]).astype(int)  # True(good)->0, False(bad)->1
else:
    # 혹시 숫자(1/2 or 0/1)로 되어 있는 경우까지 안전하게 처리
    vals = set(pd.Series(df["class"]).dropna().unique())
    if vals.issubset({1, 2}):
        # 1=good, 2=bad
        y = (df["class"] == 2).astype(int)
    elif vals.issubset({0, 1}):
        # 0=good, 1=bad 라고 가정
        y = df["class"].astype(int)
    else:
        raise ValueError(f"예상치 못한 class 값들: {vals}")

X = df.drop(columns=["class"], errors="ignore")

print("전체 분포(원본):")
print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 1) 80/20 split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain 분포:")
print(y_train.value_counts().rename({0: "good(0)", 1: "bad(1)"}))
print("\nTest 분포(현실 분포 유지):")
print(y_test.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 2) Baseline RF 학습 (선택 사항: 비교용)
# =========================
rf = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train)

# Baseline 평가
proba_base = rf.predict_proba(X_test)[:, 1]
pred_base  = (proba_base >= 0.5).astype(int)

acc_base = accuracy_score(y_test, pred_base)
auc_base = roc_auc_score(y_test, proba_base)

tn, fp, fn, tp = confusion_matrix(y_test, pred_base).ravel()
tpr_base = tp / (tp + fn) if (tp + fn) > 0 else np.nan
tnr_base = tn / (tn + fp) if (tn + fp) > 0 else np.nan

print("\n=== Baseline RF (ALL features) ===")
print(f"Accuracy     : {acc_base:.4f}")
print(f"AUC          : {auc_base:.4f}")
print(f"Sensitivity  : {tpr_base:.4f}")
print(f"Specificity  : {tnr_base:.4f}")
print(f"Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# =========================
# 3) Permutation Importance 기반 피처 선택
# =========================
from sklearn.inspection import permutation_importance

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

rf_tmp = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
)
rf_tmp.fit(X_tr, y_tr)

perm = permutation_importance(
    rf_tmp,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc"
)

importances = pd.Series(perm.importances_mean, index=X_val.columns).sort_values(ascending=False)

keep_ratio = 0.80
num_keep = max(1, int(len(importances) * keep_ratio))
selected_features = importances.index[:num_keep].tolist()

print("\nTop 15 importances:")
print(importances.head(15))
print("선택된 피처 개수:", len(selected_features))


# =========================
# 4) 선택된 피처로 최종 RF 재학습 + 평가
# =========================
rf_selected = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
)
rf_selected.fit(X_train[selected_features], y_train)

proba_sel = rf_selected.predict_proba(X_test[selected_features])[:, 1]
pred_sel  = (proba_sel >= 0.5).astype(int)

acc_sel = accuracy_score(y_test, pred_sel)
auc_sel = roc_auc_score(y_test, proba_sel)

tn, fp, fn, tp = confusion_matrix(y_test, pred_sel).ravel()
tpr_sel = tp / (tp + fn) if (tp + fn) > 0 else np.nan
tnr_sel = tn / (tn + fp) if (tn + fp) > 0 else np.nan

print("\n=== RF (Selected features) ===")
print(f"Accuracy     : {acc_sel:.4f}")
print(f"AUC          : {auc_sel:.4f}")
print(f"Sensitivity  : {tpr_sel:.4f}")
print(f"Specificity  : {tnr_sel:.4f}")
print(f"Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# =========================
# 5) ROC Curve (둘 다)
# =========================
fpr_base, tpr_base_curve, _ = roc_curve(y_test, proba_base)
fpr_sel,  tpr_sel_curve,  _ = roc_curve(y_test, proba_sel)

plt.figure()
plt.plot(fpr_base, tpr_base_curve, label=f"Baseline RF (AUC={auc_base:.4f})")
plt.plot(fpr_sel,  tpr_sel_curve,  label=f"Selected RF (AUC={auc_sel:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve (Test)")
plt.legend()
plt.show()