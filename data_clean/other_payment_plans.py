import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_property_magnitude.csv")


# 숫자 버전 새로 생성
other_payment_plans_num_map = {
    "bank": 1,
    "stores": 2,
    "none": 3,
}

df["other_payment_plans"] = df["other_payment_plans"].map(other_payment_plans_num_map)

df.to_csv("data_clean/data_clean_files/german21_other_payment_plans.csv", index=False)
