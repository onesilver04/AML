import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_personal_status.csv")


# 숫자 버전 새로 생성
other_parties_num_map = {
    "none": 1,
    "co applicant": 2,
    "guarantor": 3,
}

df["other_parties"] = df["other_parties"].map(other_parties_num_map)

df.to_csv("data_clean/data_clean_files/german21_other_parties.csv", index=False)
