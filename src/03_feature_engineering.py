"""
03_feature_engineering.py
--------------------------
Builds the model-ready feature set:
- Drops leakage-prone / redundant columns (year -> already have car_age;
  keep both but drop raw 'model' free text in favor of brand + a target-
  encoded model average, since 'model' has too many rare categories for
  one-hot).
- Creates new engineered features (price_per_cc proxy features, km_per_year,
  power_to_weight proxy, brand tier).
- One-hot encodes low-cardinality categoricals.
- Frequency/target-encodes high-cardinality 'model' column (fit on train
  only, to avoid leakage).
- Scales numeric features.
- Splits into train/test and saves as .csv + a fitted preprocessing
  pipeline (joblib) for reuse in the API.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data/processed/car_data_clean.csv"
OUT_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"

df = pd.read_csv(IN_PATH)

# ---- 1. New engineered features ----
df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)
df["power_per_cc"] = df["max_power_bhp"] / df["engine_cc"].replace(0, np.nan)
df["power_per_cc"] = df["power_per_cc"].fillna(df["power_per_cc"].median())

brand_tier_map = {
    "Bmw": "Luxury", "Mercedes-Benz": "Luxury",
    "Toyota": "Premium", "Honda": "Premium", "Kia": "Premium",
    "Ford": "Mid", "Hyundai": "Mid", "Volkswagen": "Mid",
    "Maruti Suzuki": "Economy", "Tata": "Economy",
}
df["brand_tier"] = df["brand"].map(brand_tier_map).fillna("Mid")

owner_rank = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3,
              "Fourth & Above": 4}
df["owner_rank"] = df["owner_type"].map(owner_rank)

# ---- 2. Train/test split FIRST (avoid leakage in target encoding) ----
target = "price"
feature_df = df.drop(columns=["price"])

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, df[target], test_size=0.2, random_state=42
)

# ---- 3. Target-encode 'model' using TRAIN only ----
model_price_map = y_train.groupby(X_train["model"]).mean()
global_mean = y_train.mean()

X_train["model_target_enc"] = X_train["model"].map(model_price_map)
X_test["model_target_enc"] = X_test["model"].map(model_price_map).fillna(global_mean)

# ---- 4. Drop columns not used directly as features ----
drop_cols = ["model", "year"]  # year redundant with car_age; model replaced by target enc
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# ---- 5. One-hot encode low-cardinality categoricals ----
cat_cols = ["brand", "fuel_type", "transmission", "owner_type",
            "seller_type", "location", "brand_tier"]
X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# align columns (test may be missing some dummy columns)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

# ---- 6. Scale numeric features ----
numeric_cols = ["km_driven", "mileage_kmpl", "engine_cc", "max_power_bhp",
                 "seats", "car_age", "km_per_year", "power_per_cc",
                 "model_target_enc", "owner_rank"]

scaler = StandardScaler()
X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

# Ensure consistent, sorted column order
feature_columns = sorted(X_train_enc.columns.tolist())
X_train_enc = X_train_enc[feature_columns]
X_test_enc = X_test_enc[feature_columns]

print("Final feature matrix shape:", X_train_enc.shape, X_test_enc.shape)
print("Features:", feature_columns)

# ---- 7. Save everything ----
X_train_enc.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
X_test_enc.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

joblib.dump({
    "scaler": scaler,
    "numeric_cols": numeric_cols,
    "feature_columns": feature_columns,
    "model_price_map": model_price_map,
    "global_mean": global_mean,
    "brand_tier_map": brand_tier_map,
    "owner_rank": owner_rank,
    "cat_cols": cat_cols,
}, f"{MODELS_DIR}/preprocessing_pipeline.joblib")

print("\nSaved train/test splits and preprocessing_pipeline.joblib")
