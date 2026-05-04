import os
import glob
import json
import pandas as pd

FOLDER = "SHAP/120_samples_individual_json"
files = sorted(glob.glob(os.path.join(FOLDER, "sample_*.json")))

rows = []

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "file": os.path.basename(path),
        "sample_idx": data.get("sample_idx"),
        "condition": data.get("condition"),
        "warning_type": data.get("warning_type"),
        "distance": data.get("distance"),
        "is_correct": data.get("is_correct"),
        "predicted_label": data.get("predicted_label"),
        "true_label": data.get("true_label"),
        "confidence": data.get("confidence"),
    })

df = pd.DataFrame(rows)

print(f"전체 샘플 수: {len(df)}\n")

# ---------------------------
# 1. 기본 분포
# ---------------------------
print("=== condition별 ===")
print(df["condition"].value_counts().sort_index(), "\n")

print("=== warning_type별 ===")
print(df["warning_type"].value_counts(), "\n")

print("=== distance별 ===")
print(df["distance"].value_counts(), "\n")

print("=== is_correct별 ===")
print(df["is_correct"].value_counts(), "\n")


# ---------------------------
# 2. 핵심: 4차원 조합
# ---------------------------
print("=== condition / warning_type / distance / is_correct 조합별 개수 ===")

summary = (
    df.groupby(["condition", "warning_type", "distance", "is_correct"], dropna=False)
      .size()
      .reset_index(name="count")
      .sort_values(["condition", "warning_type", "distance", "is_correct"])
)

print(summary.to_string(index=False), "\n")


# ---------------------------
# 3. 각 그룹의 sample_idx 목록
# ---------------------------
print("=== 그룹별 sample_idx 목록 ===")

sample_list = (
    df.groupby(["condition", "warning_type", "distance", "is_correct"], dropna=False)["sample_idx"]
      .apply(lambda x: sorted(x.tolist()))
      .reset_index(name="sample_indices")
      .sort_values(["condition", "warning_type", "distance", "is_correct"])
)

print(sample_list.to_string(index=False), "\n")


# ---------------------------
# 4. pivot 형태 (분석용)
# ---------------------------
print("=== pivot table (condition x distance x is_correct) ===")

pivot = pd.pivot_table(
    df,
    index=["condition", "distance"],
    columns="is_correct",
    aggfunc="size",
    fill_value=0
)

print(pivot, "\n")


# ---------------------------
# 5. CSV 저장
# ---------------------------
summary.to_csv("sample_count_summary_4d.csv", index=False, encoding="utf-8-sig")
sample_list.to_csv("sample_idx_by_group_4d.csv", index=False, encoding="utf-8-sig")

print("저장 완료:")
print("- sample_count_summary_4d.csv")
print("- sample_idx_by_group_4d.csv")