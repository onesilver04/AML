# 클래스 불균형 확인 & 원핫 인코딩으로 변경
import pandas as pd

df=pd.read_csv("german21.csv")

# 타겟
print(df["class"].value_counts())

# object 컬럼 자동 추출
cat_cols = df.select_dtypes(include="object").columns.tolist()

# one-hot
df_ohe = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=False
)

print(df_ohe.shape)
df_ohe.to_csv("german21_ohe.csv", index=False)

print(df["class"].value_counts())
