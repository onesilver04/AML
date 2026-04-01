# prefix 없는 원본 SHAP 계산 및 저장 (feature 단위)

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import shap

# =========================
# 0) 데이터 로드
# =========================
DATA_PATH = "german21_ohe.csv"
sample_indices = [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15] 
df = pd.read_csv(DATA_PATH)

# (선택) 컬럼 제거
df = df.drop(
    columns=[
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
    ],
    errors="ignore",
)

# =========================
# 타깃 설정 (bad=1, good=0)
# =========================
if "class" not in df.columns:
    raise ValueError("타깃 컬럼 'class'를 찾지 못했습니다.")

if df["class"].dtype == bool:
    # True(good)->0, False(bad)->1
    y = (~df["class"]).astype(int)
else:
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
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain 분포:")
print(y_train.value_counts().rename({0: "good(0)", 1: "bad(1)"}))
print("\nTest 분포(현실 분포 유지):")
print(y_test.value_counts().rename({0: "good(0)", 1: "bad(1)"}))

# =========================
# 2) Baseline RF 학습 (비교용)
# =========================
rf = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train)

proba_base = rf.predict_proba(X_test)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)

acc_base = accuracy_score(y_test, pred_base)
auc_base = roc_auc_score(y_test, proba_base)

tn, fp, fn, tp = confusion_matrix(y_test, pred_base).ravel()
tpr_base = tp / (tp + fn) if (tp + fn) > 0 else np.nan
tnr_base = tn / (tn + fp) if (tn + fp) > 0 else np.nan

print("\n=== Baseline RF (ALL features) ===")
print(f"Accuracy     : {acc_base:.4f}")
print(f"F1-Score     : {f1_score(y_test, pred_base):.4f}")
print(f"AUC          : {auc_base:.4f}")
print(f"Sensitivity  : {tpr_base:.4f}")
print(f"Specificity  : {tnr_base:.4f}")
print(f"Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# =========================
# 3) Permutation Importance 기반 피처 선택
# =========================
from sklearn.inspection import permutation_importance

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
pred_sel = (proba_sel >= 0.5).astype(int)

acc_sel = accuracy_score(y_test, pred_sel)
auc_sel = roc_auc_score(y_test, proba_sel)

tn, fp, fn, tp = confusion_matrix(y_test, pred_sel).ravel()
tpr_sel = tp / (tp + fn) if (tp + fn) > 0 else np.nan
tnr_sel = tn / (tn + fp) if (tn + fp) > 0 else np.nan

print("\n=== RF (Selected features) ===")
print(f"Accuracy     : {acc_sel:.4f}")
print(f"F1-Score     : {f1_score(y_test, pred_sel):.4f}")
print(f"AUC          : {auc_sel:.4f}")
print(f"Sensitivity  : {tpr_sel:.4f}")
print(f"Specificity  : {tnr_sel:.4f}")
print(f"Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# =========================
# 6) SHAP (rf_selected 기준)
# =========================
explainer = shap.TreeExplainer(rf_selected)
shap_exp = explainer(X_test[selected_features])  # shap.Explanation

# binary: (n,p,2) -> class 1만
if len(shap_exp.values.shape) == 3:
    shap_exp = shap_exp[:, :, 1]

invalid_sample_indices = [
    idx for idx in sample_indices if idx < 0 or idx >= len(X_test)
]
if invalid_sample_indices:
    raise IndexError(
        f"유효하지 않은 sample index: {invalid_sample_indices}. "
        f"가능한 범위는 0 ~ {len(X_test) - 1} 입니다."
    )

# (A) Summary - beeswarm
shap.plots.beeswarm(shap_exp, max_display=20)
plt.show()

# (B) Summary - bar
shap.plots.bar(shap_exp, max_display=20)
plt.show()

# (C) Local - waterfall
for sample_idx in sample_indices:
    print(f"\n===== Waterfall plot for sample_idx={sample_idx} =====")
    shap.plots.waterfall(shap_exp[sample_idx], max_display=20)
    plt.show()

# =========================
# 7) SHAP 중요도 (원본 feature 단위) 저장
# =========================
sv = shap_exp.values  # (n_samples, n_features)
mean_abs_shap = np.mean(np.abs(sv), axis=0)

imp_df = (
    pd.DataFrame({"feature": selected_features, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 20 SHAP feature importance (raw features):")
print(imp_df.head(20).to_string(index=False))

imp_df.to_csv("SHAP/selected_shap_feature_importance.csv", index=False)
print("\nSaved: selected_shap_feature_importance.csv")

# ==========================================================
# 8) (여기부터 추가) OHE 컬럼을 prefix로 묶어서 SHAP 중요도 재계산
#     - "샘플별 SHAP을 prefix별로 먼저 합" -> mean(|.|)
# ==========================================================
def ohe_prefix(name: str) -> str:
    # 마지막 '_' 기준으로 prefix를 잡는다 (underscore 포함 prefix 보호)
    # ex) checking_status_no checking -> checking_status
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name  # duration, age 등

feature_names = selected_features
prefix_map = {f: ohe_prefix(f) for f in feature_names}

# prefix 목록(고유)
prefixes = pd.Index([prefix_map[f] for f in feature_names])
unique_prefixes = prefixes.unique()
print("\nprefix 개수:", len(unique_prefixes))

# prefix별 컬럼 인덱스 그룹
group_indices = {}
for j, f in enumerate(feature_names):
    p = prefix_map[f]
    group_indices.setdefault(p, []).append(j)

# 샘플별 prefix SHAP 합치기
grouped_sv = np.zeros((sv.shape[0], len(unique_prefixes)), dtype=float)
for gi, p in enumerate(unique_prefixes):
    cols = group_indices[p]
    grouped_sv[:, gi] = sv[:, cols].sum(axis=1)

# prefix 중요도 = mean(abs(grouped_shap))
grouped_mean_abs = np.mean(np.abs(grouped_sv), axis=0)

# global explanation
group_imp_df = (
    pd.DataFrame({"prefix": unique_prefixes, "mean_abs_shap": grouped_mean_abs})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

def build_raw_shap_tuples(
    sv: np.ndarray,
    feature_names,
    sample_idx: int,
    top_k: int = 3,
):
    """
    sv: (n_samples, n_features) raw SHAP values
    feature_names: 원본 feature 이름 목록(selected_features)
    sample_idx: 설명할 샘플 인덱스
    top_k: 절대값 기준 상위 k개만 선택
    """
    sample_vals = sv[sample_idx]  # (n_features,)

    tuple_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sample_vals
    })

    tuple_df["abs_shap"] = tuple_df["shap_value"].abs()
    tuple_df["direction"] = np.where(
        tuple_df["shap_value"] >= 0,
        "increase_risk",
        "decrease_risk"
    )

    tuple_df = (
        tuple_df
        .sort_values("abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return tuple_df

top_k = 3

# =========================
# 🔥 Raw SHAP 전체 + Top 10 출력
# =========================

def print_raw_shap_full(
    sv: np.ndarray,
    feature_names,
    sample_idx: int,
    top_k: int = 10,
):
    sample_vals = sv[sample_idx]

    df_full = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sample_vals
    })

    df_full["abs_shap"] = df_full["shap_value"].abs()
    df_full["direction"] = np.where(
        df_full["shap_value"] >= 0,
        "increase_risk",
        "decrease_risk"
    )

    df_full = df_full.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("\n===== 🔍 Full Raw SHAP (sorted) =====\n")
    print(df_full.to_string(index=False))

    print("\n===== 🔝 Top 10 Raw SHAP =====\n")
    print(df_full.head(top_k).to_string(index=False))


# save tuples
def save_shap_tuples_json(
    tuple_df, 
    sample_idx, 
    prediction_label, 
    predict_proba=None, 
    save_path="SHAP/shap_tuples_non_prefix.json"):
    
    records = []

    for _, row in tuple_df.iterrows():
        records.append({
            "feature": row["feature"],
            "shap_value": float(row["shap_value"]),
            "abs_shap": float(row["abs_shap"]),
            "direction": row["direction"]
        })

    data = {
        "sample_idx": sample_idx,
        "prediction": {
            "label": prediction_label,
            "probability": predict_proba
        },
        "tuples": records
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {save_path}")
    

    
print("\nTop 20 SHAP importance (Grouped by prefix):")
print(group_imp_df.head(20).to_string(index=False))

group_imp_df.to_csv("SHAP/selected_shap_prefix_importance.csv", index=False)
print("\nSaved: selected_shap_prefix_importance.csv")

for sample_idx in sample_indices:
    sample_tuple_df = build_raw_shap_tuples(
        sv=sv,
        feature_names=selected_features,
        sample_idx=sample_idx,
        top_k=top_k,
    )

    print_raw_shap_full(
        sv=sv,
        feature_names=selected_features,
        sample_idx=sample_idx,
        top_k=10,
    )

    print(f"\nSample-specific raw SHAP tuples (sample_idx={sample_idx}):")
    print(sample_tuple_df.to_string(index=False))

    prediction_label = (
        "BAD CREDIT RISK" if pred_sel[sample_idx] == 1 else "GOOD CREDIT RISK"
    )
    prediction_proba = float(proba_sel[sample_idx])

    save_shap_tuples_json(
        sample_tuple_df,
        sample_idx,
        prediction_label=prediction_label,
        predict_proba=prediction_proba,
        save_path=f"SHAP/Feature Importance/shap_tuples_non_prefix_{sample_idx}.json",
    )
