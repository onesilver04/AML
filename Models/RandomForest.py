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

# (선택) 외국인 컬럼 제거: 네가 이미 제거했다고 했지만, 파일에 남아있으면 제거
df = df.drop(columns=["foreign_worker_no", "foreign_worker_yes"], errors="ignore")

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
# 2) RandomForest 학습
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
# 3) 예측 & 메트릭
# =========================
proba = rf.predict_proba(X_test)

# 안전장치: 혹시라도 한 클래스만 학습되면(이론상 stratify면 거의 안 생김) 방어
if proba.shape[1] == 2:
    y_proba = proba[:, 1]  # bad(1) 확률
else:
    only_class = rf.classes_[0]
    y_proba = np.zeros(len(X_test)) if only_class == 0 else np.ones(len(X_test))

y_pred = (y_proba >= 0.5).astype(int)  # threshold=0.5

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
# 4) ROC Curve
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