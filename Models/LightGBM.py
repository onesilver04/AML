import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier


# =========================
# 0) 데이터 로드
# =========================
DATA_PATH = "german21_ohe.csv"   # 원핫 완료 파일
df = pd.read_csv(DATA_PATH)

# 타깃 설정: class_bad(1=bad) 우선 사용
if "class_bad" in df.columns:
    y = df["class_bad"].astype(int)  # 1=bad, 0=good
    X = df.drop(columns=["class_bad", "class_good"], errors="ignore")
else:
    # class가 1(good), 2(bad)로 남아있는 경우
    y = (df["class"] == 2).astype(int)
    X = df.drop(columns=["class"], errors="ignore")

print("전체 분포(원본):")
print(y.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 1) 80/20 split (원본 분포 유지)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTest 분포(현실 분포 유지):")
print(y_test.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 2) Train에서만 random under-sampling
# =========================
train_df = pd.concat([X_train, y_train.rename("class_bad")], axis=1)

train_good = train_df[train_df["class_bad"] == 0]
train_bad  = train_df[train_df["class_bad"] == 1]

train_good_down = resample(
    train_good,
    replace=False,
    n_samples=len(train_bad),
    random_state=42
)

train_balanced = pd.concat([train_good_down, train_bad]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

X_train_bal = train_balanced.drop(columns=["class_bad"])
y_train_bal = train_balanced["class_bad"].astype(int)

print("\nTrain 분포(언더샘플링 후):")
print(y_train_bal.value_counts().rename({0: "good(0)", 1: "bad(1)"}))


# =========================
# 3) LGBM 학습
# =========================
lgbm = LGBMClassifier(
    n_estimators=2000,          # 충분히 크게 두고 early stopping으로 제어
    learning_rate=0.02,
    num_leaves=31,
    max_depth=-1,               # 제한 없음
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    n_jobs=-1
)

# early stopping을 위해 validation을 따로 나눔(Train balanced 안에서)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_bal, y_train_bal, test_size=0.2, random_state=42, stratify=y_train_bal
)

lgbm.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[],
)

# =========================
# 4) 예측 & 메트릭
# =========================
y_proba = lgbm.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

print("\n=== Metrics (LightGBM) on ORIGINAL test distribution ===")
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
plt.plot(fpr, tpr, label=f"LightGBM (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve (LightGBM, Test: original distribution)")
plt.legend()
plt.show()
