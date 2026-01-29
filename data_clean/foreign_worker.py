import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_own_telephone.csv")


# 숫자 버전 새로 생성
foreign_worker_num_map = {
    "yes": 1,
    "no": 2,
}

df["foreign_worker"] = df["foreign_worker"].map(foreign_worker_num_map)

df.to_csv("data_clean/data_clean_files/german21_foreign_worker.csv", index=False)
