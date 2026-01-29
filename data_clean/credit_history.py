import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_checking_status.csv")


# 숫자 버전 새로 생성
credit_history_num_map = {
    "no credits/all paid": 1,
    "all paid": 2,
    "existing paid": 3,
    "delayed previously": 4,
    "critical/other existing credit": 5
}

df["credit_history"] = df["credit_history"].map(credit_history_num_map)

df.to_csv("data_clean/data_clean_files/german21_credit_history.csv", index=False)

