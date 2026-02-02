import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

import matplotlib.pyplot as plt


# =========================
# 0) 데이터 로드
# =========================
DATA_PATH = "german21_ohe.csv"

df = pd.read_csv(DATA_PATH)

# 타깃 설정 (bad=1, good=0)
if "class_bad" in df.columns:
    y = df["class_bad"].astype(int)
    X = df.drop(columns=["class_bad", "class_good"], errors="ignore")  # 누수 방지
elif "class" in df.columns:
    # class: 1=good, 2=bad
    y = (df["class"] == 2).astype(int)
    X = df.drop(columns=["class"], errors="ignore")
else:
    raise ValueError("타깃 컬럼을 찾지 못했습니다. (class_bad/class_good 또는 class가 필요)")

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

print("\nTest 분포(현실 분포 유지):")
print(y_test.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# # =========================
# # 2) Train에서만 Random Under-Sampling
# # =========================
# train_df = pd.concat([X_train, y_train.rename("class_bad")], axis=1)

# train_good = train_df[train_df["class_bad"] == 0]
# train_bad  = train_df[train_df["class_bad"] == 1]

# train_good_down = resample(
#     train_good,
#     replace=False,
#     n_samples=len(train_bad),
#     random_state=42
# )

# train_balanced = pd.concat([train_good_down, train_bad]).sample(
#     frac=1, random_state=42
# ).reset_index(drop=True)

# X_train_bal = train_balanced.drop(columns=["class_bad"])
# y_train_bal = train_balanced["class_bad"].astype(int)

# print("\nTrain 분포(언더샘플링 후):")
# print(y_train_bal.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 3) RandomForest 학습
# =========================
rf = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
)

rf.fit(X_train, y_train)


# =========================
# 4) 예측 & 메트릭
# =========================
y_proba = rf.predict_proba(X_test)[:, 1]   # bad(1) 확률
y_pred  = (y_proba >= 0.5).astype(int)    # 기본 threshold=0.5

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # TPR
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # TNR

print("\n=== Metrics (RandomForest) on ORIGINAL test distribution ===")
print(f"Accuracy     : {acc:.4f}")
print(f"AUC          : {auc:.4f}")
print(f"Sensitivity  : {sensitivity:.4f} (TPR, Recall of bad=1)")
print(f"Specificity  : {specificity:.4f} (TNR, Recall of good=0)")
print(f"Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# =========================
# 5) ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve (RandomForest, Test: original distribution)")
plt.legend()
plt.show()

joblib.dump(rf, "rf_model.joblib")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)