"""
02_eda.py
---------
Exploratory Data Analysis on the cleaned dataset.
Saves summary stats (CSV) and key plots (PNG) to plots/.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data/processed/car_data_clean.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

df = pd.read_csv(IN_PATH)

# 1. Summary statistics
summary = df.describe(include="all").transpose()
summary.to_csv(f"{PLOTS_DIR}/summary_statistics.csv")
print(summary)

# 2. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50, kde=True, color="#2E86AB")
plt.title("Distribution of Used Car Prices")
plt.xlabel("Price (USD)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_price_distribution.png", dpi=140)
plt.close()

# 3. Price vs car age
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="car_age", y="price", alpha=0.3, color="#F18F01")
plt.title("Price vs Car Age")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_price_vs_age.png", dpi=140)
plt.close()

# 4. Price vs km_driven
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="price", alpha=0.3, color="#C73E1D")
plt.title("Price vs Kilometers Driven")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_price_vs_km.png", dpi=140)
plt.close()

# 5. Average price by brand
plt.figure(figsize=(10, 6))
order = df.groupby("brand")["price"].mean().sort_values(ascending=False).index
sns.barplot(data=df, x="brand", y="price", order=order, errorbar=None,
            palette="viridis")
plt.title("Average Price by Brand")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_avg_price_by_brand.png", dpi=140)
plt.close()

# 6. Price by fuel type / transmission
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="fuel_type", y="price", ax=axes[0], palette="Set2")
axes[0].set_title("Price by Fuel Type")
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df, x="transmission", y="price", ax=axes[1], palette="Set3")
axes[1].set_title("Price by Transmission")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_price_by_fuel_transmission.png", dpi=140)
plt.close()

# 7. Correlation heatmap (numeric columns)
num_df = df.select_dtypes(include="number")
plt.figure(figsize=(9, 7))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_correlation_heatmap.png", dpi=140)
plt.close()

print("\nEDA plots saved to plots/")
