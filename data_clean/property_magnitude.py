import pandas as pd

df = pd.read_csv("data_clean/data_clean_files/german21_other_parties.csv")


# 숫자 버전 새로 생성
property_magnitude_num_map = {
    "real estate": 1,
    "life insurance": 2,
    "car": 3,
    "no known property": 4,
}

df["property_magnitude"] = df["property_magnitude"].map(property_magnitude_num_map)

df.to_csv("data_clean/data_clean_files/german21_property_magnitude.csv", index=False)
