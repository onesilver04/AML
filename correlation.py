import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# (필요하면 주석 해제해서 Agg 백엔드 사용)
# os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
# matplotlib.use("Agg")

data_path = "german21.csv"
df = pd.read_csv(data_path)

print(f"데이터 shape: {df.shape}")

# 1) 수치형 피처만 선택 (이상치 처리/스케일링 없이 그대로 사용)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
print(f"수치형 피처 ({len(numeric_cols)}개): {numeric_cols}")

num_df = df[numeric_cols]

# 2) 피어슨 상관계수 행렬 계산
corr = num_df.corr(method="pearson")
print("\n피어슨 상관계수 행렬:")
print(corr)

# 4) 히트맵으로 시각화 (선택)
plt.figure(figsize=(8, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("correlation_numeric_heatmap.png", dpi=160)
plt.show()