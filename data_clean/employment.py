import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_savings_status.csv")


# 숫자 버전 새로 생성
employment_num_map = {
    "unemployed": 1,
    "<1": 2,
    "1<=X<4": 3,
    "4<=X<7": 4,
    ">=7": 5
}

df["employment"] = df["employment"].map(employment_num_map)

df.to_csv("data_clean/data_clean_files/german21_employment.csv", index=False)
