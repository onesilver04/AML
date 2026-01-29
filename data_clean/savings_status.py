import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_purpose.csv")


# 숫자 버전 새로 생성
savings_status_num_map = {
    "<100": 1,
    "100<=X<500": 2,
    "500<=X<1000": 3,
    ">=1000": 4,
    "no known savings": 5
}

df["savings_status"] = df["savings_status"].map(savings_status_num_map)

df.to_csv("data_clean/data_clean_files/german21_savings_status.csv", index=False)

