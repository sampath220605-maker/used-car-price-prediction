#!/usr/bin/env bash
# Runs the entire pipeline end-to-end, in order:
# Dataset -> Cleaning -> EDA -> Feature Engineering -> 5 Models ->
# Comparison -> Tuning -> Best Model -> Explainability
#
# Usage:
#   bash run_pipeline.sh
set -e

cd "$(dirname "$0")"

echo "== [1/6] Generating dataset (skip this step if you have your own CSV) =="
python3 src/00_generate_dataset.py

echo "== [2/6] Cleaning data =="
python3 src/01_data_cleaning.py

echo "== [3/6] Running EDA =="
python3 src/02_eda.py

echo "== [4/6] Feature engineering =="
python3 src/03_feature_engineering.py

echo "== [5/6] Training & comparing 5 models =="
python3 src/04_train_models.py

echo "== [6/6] Hyperparameter tuning + explainability =="
python3 src/05_hyperparameter_tuning.py
python3 src/06_explainability.py

echo ""
echo "Pipeline complete. Best model saved to models/best_model.joblib"
echo "Start the API with:  uvicorn api.app:app --reload --port 8000"
echo "Then open web/index.html in a browser."
