import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_other_payment_plans.csv")


# 숫자 버전 새로 생성
housing_num_map = {
    "rent": 1,
    "own": 2,
    "for free": 3,
}

df["housing"] = df["housing"].map(housing_num_map)

df.to_csv("data_clean/data_clean_files/german21_housing.csv", index=False)
