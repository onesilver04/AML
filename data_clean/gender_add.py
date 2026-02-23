import pandas as pd

df_ohe = pd.read_csv("german21_ohe.csv")
df_kaggle = pd.read_csv("Kaggle/german_credit_data (1).csv")

sex_col = df_kaggle["Sex"]

df_ohe["Sex"] = sex_col

df_ohe["Sex"] = df_ohe["Sex"].map({"male":True, "female":False})

df_ohe.to_csv("german21_ohe_with_sex.csv", index=False)