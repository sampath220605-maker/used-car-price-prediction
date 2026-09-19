"""
04_train_models.py
-------------------
Trains 5 regression models on the engineered features and compares them
using RMSE, MAE, R2, and 5-fold CV R2. Saves a comparison table (CSV) and
bar-chart plot, plus every trained model (joblib) for later use.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1),
}

results = []
fitted_models = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    results.append({
        "Model": name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "CV_R2_mean": round(cv_scores.mean(), 4),
        "CV_R2_std": round(cv_scores.std(), 4),
        "Train_Time_s": round(train_time, 2),
    })
    fitted_models[name] = model
    joblib.dump(model, f"{MODELS_DIR}/{name.replace(' ', '_').lower()}.joblib")
    print(f"{name:20s} | RMSE={rmse:9.2f} | MAE={mae:9.2f} | R2={r2:.4f} "
          f"| CV_R2={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)
print("\n", results_df.to_string(index=False))

# ---- Comparison plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = results_df["Model"]

axes[0].barh(order, results_df["R2"], color="#2E86AB")
axes[0].set_xlabel("R2 Score (higher is better)")
axes[0].set_title("Model Comparison: R2")
axes[0].invert_yaxis()

axes[1].barh(order, results_df["RMSE"], color="#C73E1D")
axes[1].set_xlabel("RMSE (lower is better)")
axes[1].set_title("Model Comparison: RMSE")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_model_comparison.png", dpi=140)
plt.close()

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by R2: {best_model_name}")
with open(f"{MODELS_DIR}/best_model_name.txt", "w") as f:
    f.write(best_model_name)
