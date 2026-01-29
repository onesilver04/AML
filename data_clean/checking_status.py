import pandas as pd

df = pd.read_csv("german21.csv")


# 숫자 버전 새로 생성
checking_status_num_map = {
    "<0": 1,
    "0<=X<200": 2,
    ">=200": 3,
    "no checking": 4
}

df["checking_status"] = df["checking_status"].map(checking_status_num_map)

df.to_csv("german21_checking_status.csv", index=False)

