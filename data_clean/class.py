import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_foreign_worker.csv")


# 숫자 버전 새로 생성
class_num_map = {
    "good": 1,
    "bad": 2,
}

df["class"] = df["class"].map(class_num_map)

df.to_csv("data_clean/data_clean_files/german21_class.csv", index=False)
