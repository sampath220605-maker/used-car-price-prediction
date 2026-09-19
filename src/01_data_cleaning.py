"""
01_data_cleaning.py
--------------------
Loads the raw dataset and produces a cleaned version:
- Fixes inconsistent casing/whitespace in categorical columns
- Strips units embedded in numeric-as-string columns
- Handles missing values (median/mode imputation)
- Removes duplicate rows
- Caps/removes extreme outliers (IQR-based)
- Saves cleaned dataset to data/processed/car_data_clean.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/car_data.csv"
OUT_PATH = PROJECT_ROOT / "data/processed/car_data_clean.csv"
LOG_PATH = PROJECT_ROOT / "data/processed/cleaning_report.txt"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


df = pd.read_csv(RAW_PATH)
log(f"Raw shape: {df.shape}")

# 1. Normalize categorical text columns
cat_cols = ["brand", "model", "fuel_type", "transmission", "owner_type",
            "seller_type", "location"]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# 2. Strip embedded units from numeric-as-string columns
def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).lower().replace("kmpl", "").replace("bhp", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in ["mileage_kmpl", "max_power_bhp"]:
    df[col] = df[col].apply(to_float)

# 3. Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
log(f"Dropped {before - len(df)} duplicate rows")

# 4. Missing value report + imputation
log("\nMissing values before imputation:")
log(df.isna().sum()[df.isna().sum() > 0].to_string())

num_cols = ["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
for col in num_cols:
    median = df[col].median()
    df[col] = df[col].fillna(median)

for col in ["fuel_type"]:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

log("\nMissing values after imputation:")
log(str(df.isna().sum().sum()) + " total missing cells remain")

# 5. Fix impossible / outlier values
# km_driven: cap using IQR
for col in ["km_driven", "price"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    log(f"{col}: capping {n_outliers} outliers to [{lower:.0f}, {upper:.0f}]")
    df[col] = df[col].clip(lower=max(lower, 0), upper=upper)

# seats sanity (2-9)
df["seats"] = df["seats"].clip(2, 9).round().astype(int)

# year sanity
df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]

# 6. Derived sanity column: car_age
df["car_age"] = 2026 - df["year"]

df = df.reset_index(drop=True)
log(f"\nFinal cleaned shape: {df.shape}")

df.to_csv(OUT_PATH, index=False)
with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

log(f"\nSaved cleaned dataset -> {OUT_PATH}")
