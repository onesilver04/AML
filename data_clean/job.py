import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_housing.csv")


# 숫자 버전 새로 생성
job_num_map = {
    "unemp/unskilled non res": 1,
    "unskilled resident": 2,
    "skilled": 3,
    "high qualif/self emp/mgmt": 4
}

df["job"] = df["job"].map(job_num_map)

df.to_csv("data_clean/data_clean_files/german21_job.csv", index=False)
