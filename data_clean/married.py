# # 결혼 여부를 판단하는 컬럼 생성
# import pandas as pd

# df = pd.read_csv("german21_ohe.csv")
# df["Married"] = (
#     df["personal_status_male mar/wid"] |
#     df["personal_status_female div/dep/mar"]
# )

# df = df.drop(columns=[
#     "personal_status_male single",
#     "personal_status_male mar/wid",
#     "personal_status_male div/sep",
#     "personal_status_female div/dep/mar"
# ], errors="ignore")

# # bool 유지
# df["Married"] = df["Married"].astype(bool)

# # 새 파일 저장
# df.to_csv("german21_ohe_with_married.csv", index=False)

import pandas as pd

df_ohe = pd.read_csv("german21_ohe.csv")
df_mar = pd.read_csv("german21_ohe_with_married.csv")

sex_col = df_mar["Married"]

df_ohe["Married"] = sex_col

df_ohe["Married"] = df_ohe["Married"].astype(bool)

df_ohe.to_csv("german21_ohe.csv", index=False)