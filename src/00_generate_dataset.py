"""
00_generate_dataset.py
-----------------------
Generates a realistic synthetic 'used car' dataset since no raw dataset
was supplied by the user. If you have your own CSV (e.g. from Kaggle's
"Used Car Price Prediction" datasets), just drop it in data/raw/car_data.csv
with similar columns and skip this script.

Columns produced:
    brand, model, year, km_driven, fuel_type, transmission, owner_type,
    seller_type, mileage_kmpl, engine_cc, max_power_bhp, seats, location,
    price  (target, in USD)

Includes intentional messiness (missing values, duplicates, inconsistent
casing/units, outliers) so the cleaning step has real work to do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

rng = np.random.default_rng(42)

N = 8000

brands_models = {
    "Toyota": ["Corolla", "Camry", "RAV4", "Yaris", "Innova"],
    "Honda": ["Civic", "City", "Accord", "Jazz", "CR-V"],
    "Ford": ["Focus", "Fiesta", "EcoSport", "Figo", "Endeavour"],
    "Hyundai": ["i10", "i20", "Creta", "Verna", "Santro"],
    "Maruti Suzuki": ["Swift", "Baleno", "Dzire", "WagonR", "Alto"],
    "BMW": ["3 Series", "5 Series", "X1", "X5"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLA"],
    "Volkswagen": ["Polo", "Vento", "Tiguan"],
    "Tata": ["Nexon", "Tiago", "Harrier"],
    "Kia": ["Seltos", "Sonet", "Carnival"],
}
brand_list = list(brands_models.keys())

fuel_types = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
fuel_probs = [0.45, 0.35, 0.08, 0.05, 0.07]
transmissions = ["Manual", "Automatic"]
owner_types = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"]
owner_probs = [0.55, 0.28, 0.12, 0.05]
seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
locations = ["Hyderabad", "Mumbai", "Delhi", "Bangalore", "Chennai",
             "Pune", "Kolkata", "Ahmedabad"]

# base price per brand (proxy for brand prestige/value retention)
brand_base_price = {
    "Toyota": 18000, "Honda": 16000, "Ford": 13000, "Hyundai": 12000,
    "Maruti Suzuki": 9000, "BMW": 38000, "Mercedes-Benz": 42000,
    "Volkswagen": 15000, "Tata": 10000, "Kia": 14000,
}

current_year = 2026

rows = []
for i in range(N):
    brand = rng.choice(brand_list)
    model = rng.choice(brands_models[brand])
    year = int(rng.integers(2005, 2026))
    age = current_year - year
    km_driven = max(500, int(rng.normal(15000 * max(age, 1), 12000)))
    fuel = rng.choice(fuel_types, p=fuel_probs)
    transmission = rng.choice(transmissions, p=[0.68, 0.32])
    owner = rng.choice(owner_types, p=owner_probs)
    seller = rng.choice(seller_types, p=[0.55, 0.35, 0.10])
    location = rng.choice(locations)

    mileage = round(float(rng.normal(18, 4)), 1)
    engine_cc = int(rng.choice([800, 1000, 1200, 1500, 1600, 2000, 2500, 3000]))
    power = round(float(max(35, rng.normal(engine_cc / 15, 15))), 1)
    seats = int(rng.choice([4, 5, 5, 5, 7], p=[0.05, 0.55, 0.0, 0.0, 0.4]) if False else rng.choice([4, 5, 7], p=[0.05, 0.75, 0.20]))

    base = brand_base_price[brand]
    depreciation = 0.92 ** age
    km_penalty = 1 - min(km_driven / 400000, 0.6)
    trans_bonus = 1.08 if transmission == "Automatic" else 1.0
    owner_penalty = {"First Owner": 1.0, "Second Owner": 0.9,
                      "Third Owner": 0.8, "Fourth & Above": 0.7}[owner]
    fuel_bonus = {"Petrol": 1.0, "Diesel": 1.05, "CNG": 0.95,
                  "Electric": 1.25, "Hybrid": 1.15}[fuel]

    price = (base * depreciation * km_penalty * trans_bonus *
              owner_penalty * fuel_bonus * (1 + power / 500))
    price = price * float(rng.normal(1.0, 0.08))  # market noise
    price = max(800, round(price, -1))  # round to nearest 10, floor

    rows.append([
        brand, f"{brand} {model}", year, km_driven, fuel, transmission,
        owner, seller, mileage, engine_cc, power, seats, location, price
    ])

df = pd.DataFrame(rows, columns=[
    "brand", "model", "year", "km_driven", "fuel_type", "transmission",
    "owner_type", "seller_type", "mileage_kmpl", "engine_cc",
    "max_power_bhp", "seats", "location", "price"
])

# ---- inject realistic messiness ----

# 1. Missing values
for col, frac in [("mileage_kmpl", 0.05), ("max_power_bhp", 0.03),
                   ("seats", 0.02), ("engine_cc", 0.02), ("fuel_type", 0.01)]:
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2. Inconsistent casing / whitespace in categoricals
def messy_case(x):
    r = rng.random()
    if r < 0.1:
        return str(x).upper()
    elif r < 0.2:
        return str(x).lower()
    elif r < 0.25:
        return f" {x} "
    return x

for col in ["fuel_type", "transmission", "seller_type"]:
    df[col] = df[col].apply(messy_case)

# 3. Units mixed into mileage/power as strings for a subset of rows
str_idx = rng.choice(df.index, size=int(len(df) * 0.08), replace=False)
df["mileage_kmpl"] = df["mileage_kmpl"].astype(object)
df.loc[str_idx, "mileage_kmpl"] = df.loc[str_idx, "mileage_kmpl"].apply(
    lambda x: f"{x} kmpl" if pd.notna(x) else x)

str_idx2 = rng.choice(df.index, size=int(len(df) * 0.08), replace=False)
df["max_power_bhp"] = df["max_power_bhp"].astype(object)
df.loc[str_idx2, "max_power_bhp"] = df.loc[str_idx2, "max_power_bhp"].apply(
    lambda x: f"{x} bhp" if pd.notna(x) else x)

# 4. Outliers
out_idx = rng.choice(df.index, size=20, replace=False)
df.loc[out_idx, "price"] = df.loc[out_idx, "price"] * rng.uniform(8, 15)
out_idx2 = rng.choice(df.index, size=15, replace=False)
df.loc[out_idx2, "km_driven"] = rng.integers(500000, 900000, size=15)

# 5. Duplicates
dupes = df.sample(60, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 6. Shuffle
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv(PROJECT_ROOT / "data/raw/car_data.csv", index=False)
print("Saved raw dataset:", df.shape)
print(df.head())
