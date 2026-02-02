import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("german21_ohe.csv")

# 1) 타깃 분포 확인
print("before:")
print(df["class_bad"].value_counts())

# 2) good / bad 분리
df_good = df[df["class_bad"] == 0]
df_bad  = df[df["class_bad"] == 1]

# 3) good을 bad 개수만큼 랜덤으로 다운샘플링
df_good_down = resample(
    df_good,
    replace=False,
    n_samples=len(df_bad),
    random_state=42
)

# 4) 합치고 섞기
df_balanced = pd.concat([df_good_down, df_bad], axis=0).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print("\nafter:")
print(df_balanced["class_bad"].value_counts())

# 5) 저장
df_balanced.to_csv("german21_ohe_balanced.csv", index=False)
print("\nsaved: german21_ohe_balanced.csv")
print("shape:", df_balanced.shape)
