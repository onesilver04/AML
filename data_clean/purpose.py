import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_credit_history.csv")


# 숫자 버전 새로 생성
purpose_num_map = {
    "new car": 1,
    "used car": 2,
    "furniture/equipment": 3,
    "radio/tv": 4,
    "domestic appliance": 5,
    "repairs": 6,
    "education": 7,
    "retraining": 8,
    "business": 9,
    "other": 10
}

df["purpose"] = df["purpose"].map(purpose_num_map)

df.to_csv("data_clean/data_clean_files/german21_purpose.csv", index=False)


print(df["purpose"].value_counts())

