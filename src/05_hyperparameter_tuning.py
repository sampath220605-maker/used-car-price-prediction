"""
05_hyperparameter_tuning.py
----------------------------
Tunes the best-performing model from step 4 (selected automatically via
models/best_model_name.txt) using RandomizedSearchCV, then re-evaluates
on the held-out test set and saves the final tuned model as
models/best_model.joblib.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

with open(f"{MODELS_DIR}/best_model_name.txt") as f:
    best_model_name = f.read().strip()

print(f"Tuning: {best_model_name}")

if best_model_name == "XGBoost":
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
elif best_model_name == "Random Forest":
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [8, 10, 14, 18, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
else:
    raise ValueError(f"No tuning grid defined for {best_model_name}; "
                      f"add one or fall back to the step-4 model as-is.")

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=40, cv=5,
    scoring="r2", random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print("\nBest params:", search.best_params_)
print("Best CV R2:", search.best_score_)

best_model = search.best_estimator_
preds = best_model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
mae = float(mean_absolute_error(y_test, preds))
r2 = float(r2_score(y_test, preds))

print(f"\nTuned {best_model_name} on TEST set:")
print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")

joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib")

with open(f"{MODELS_DIR}/best_model_metrics.json", "w") as f:
    json.dump({
        "model_name": best_model_name,
        "best_params": search.best_params_,
        "cv_r2": search.best_score_,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }, f, indent=2)

print("\nSaved tuned model -> models/best_model.joblib")
