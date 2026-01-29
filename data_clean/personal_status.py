import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_employment.csv")


# 숫자 버전 새로 생성
personal_status_num_map = {
    "male div/sep": 1,
    "female div/dep/mar": 2,
    "male single": 3,
    "male mar/wid": 4,
    "female single": 5
}

df["personal_status"] = df["personal_status"].map(personal_status_num_map)

df.to_csv("data_clean/data_clean_files/german21_personal_status.csv", index=False)
