"""
06_explainability.py
---------------------
Explains the tuned best model using:
- Built-in feature importances (bar chart)
- SHAP values: global summary (beeswarm) + a single-prediction
  waterfall/force-style explanation for one example row.
Saves everything as PNGs in plots/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

model = joblib.load(f"{MODELS_DIR}/best_model.joblib")

# ---- 1. Native feature importance ----
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 7))
importances.sort_values().plot.barh(color="#2E86AB")
plt.title("Top 15 Feature Importances (Tuned Model)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_feature_importance.png", dpi=140)
plt.close()
print("Top features:\n", importances)

# ---- 2. SHAP global explanation ----
sample = X_test.sample(min(500, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_shap_summary.png", dpi=140, bbox_inches="tight")
plt.close()

# ---- 3. SHAP single-prediction waterfall ----
row_idx = 0
plt.figure()
shap.plots.waterfall(shap_values[row_idx], show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_shap_waterfall_example.png", dpi=140, bbox_inches="tight")
plt.close()

print("\nSaved feature importance + SHAP plots to plots/")
