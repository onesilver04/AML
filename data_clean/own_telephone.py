import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_job.csv")


# 숫자 버전 새로 생성
own_telephone_num_map = {
    "none": 1,
    "yes": 2,
}

df["own_telephone"] = df["own_telephone"].map(own_telephone_num_map)

df.to_csv("data_clean/data_clean_files/german21_own_telephone.csv", index=False)
