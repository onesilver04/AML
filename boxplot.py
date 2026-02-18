import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("german21.csv")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

n = len(num_cols)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
axes = axes.flatten()

for i, c in enumerate(num_cols):
    axes[i].boxplot(df[c].dropna(), vert=True)
    axes[i].set_title(c)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("boxplot_numeric_grid.png", dpi=160)
plt.show()