# Used Car Price Prediction System

An end-to-end data science project that predicts the resale price of a used
car, from raw data to a deployed web app.

```
Dataset → Cleaning → EDA → Feature Engineering → 5 Models → Comparison
→ Tuning → Best Model → Explainability → API → Web UI → Deployment
```

## 1. Project structure

```
used_car_price_prediction/
├── data/
│   ├── raw/car_data.csv              # raw (synthetic) dataset, intentionally messy
│   └── processed/                    # cleaned data, train/test splits, comparison tables
├── src/                               # pipeline scripts, run in order
│   ├── 00_generate_dataset.py         # synthetic data generator (skip if you supply your own CSV)
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_models.py             # trains & compares 5 models
│   ├── 05_hyperparameter_tuning.py    # tunes the best model
│   └── 06_explainability.py           # feature importance + SHAP
├── plots/                             # all EDA / comparison / SHAP charts (PNG)
├── models/                            # saved models + preprocessing pipeline (joblib)
├── api/
│   └── app.py                         # FastAPI inference service
├── web/
│   └── index.html                     # single-page web UI that calls the API
├── deployment/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml             # runs API (8000) + web UI (8080) together
├── run_pipeline.sh                    # runs the whole ML pipeline end-to-end
└── README.md
```

## 2. Dataset

No dataset was supplied, so `src/00_generate_dataset.py` generates a
realistic **synthetic** used-car dataset (8,000 rows) with the kind of mess
real data has: missing values, inconsistent casing, units embedded in
numbers (`"18.5 kmpl"`), duplicates, and outliers. Price is a function of
brand, age, mileage, ownership, fuel type, transmission and engine power
plus market noise, so the ML pipeline has real signal to learn.

**To use your own data instead:** drop a CSV with similar columns
(`brand, model, year, km_driven, fuel_type, transmission, owner_type,
seller_type, mileage_kmpl, engine_cc, max_power_bhp, seats, location,
price`) into `data/raw/car_data.csv` and skip step 00 in the pipeline.

## 3. How to run everything

```bash
cd used_car_price_prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/requirements.txt
bash run_pipeline.sh
```

On Windows PowerShell, create and use the virtual environment with:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r deployment\requirements.txt
```

Run the pipeline and API with the same environment that created the model
artifacts. This avoids XGBoost model compatibility errors between different
Python or XGBoost installations.

This runs cleaning → EDA → feature engineering → trains 5 models → tunes
the best one → produces explainability plots. All outputs land in
`data/processed/`, `plots/`, and `models/`.

Then start the API:

```bash
source .venv/bin/activate
python -m uvicorn api.app:app --reload --port 8000
```

On Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m uvicorn api.app:app --reload --port 8000
```

Open `web/index.html` in a browser (or serve it: `python -m http.server 8080`
from inside `web/`) — it calls `http://127.0.0.1:8000/predict`.

**Or run everything in Docker:**

```bash
docker compose -f deployment/docker-compose.yml up --build
```
- API → http://localhost:8000/docs
- Web UI → http://localhost:8080

## 4. Pipeline details

### Cleaning (`01_data_cleaning.py`)
- Normalizes categorical text (casing/whitespace)
- Strips embedded units (`"110 bhp"` → `110.0`)
- Median/mode-imputes missing values
- Drops duplicate rows
- Caps outliers using the IQR method (3×IQR bounds)

### EDA (`02_eda.py`)
Produces `plots/01`–`06`: price distribution, price vs. age, price vs.
km driven, average price by brand, price by fuel/transmission, and a
correlation heatmap.

### Feature engineering (`03_feature_engineering.py`)
- New features: `car_age`, `km_per_year`, `power_per_cc`, `brand_tier`,
  `owner_rank`
- Target-encodes the high-cardinality `model` column (fit on train only,
  to avoid leakage)
- One-hot encodes low-cardinality categoricals
- Standard-scales numeric features
- Saves a `preprocessing_pipeline.joblib` so the API can transform raw
  input the same way at inference time

### 5 models trained & compared (`04_train_models.py`)
Linear Regression, Ridge Regression, Decision Tree, Random Forest, XGBoost —
evaluated with RMSE, MAE, R², and 5-fold CV R² (`plots/07_model_comparison.png`).

| Model | RMSE | MAE | R² | CV R² |
|---|---|---|---|---|
| **XGBoost** | **1288.85** | **613.99** | **0.9704** | 0.9541 |
| Random Forest | 1414.02 | 695.16 | 0.9644 | 0.9474 |
| Decision Tree | 2471.85 | 1013.76 | 0.8911 | 0.8964 |
| Linear Regression | 3330.28 | 2482.51 | 0.8024 | 0.7859 |
| Ridge Regression | 3330.95 | 2482.29 | 0.8023 | 0.7859 |

### Tuning (`05_hyperparameter_tuning.py`)
`RandomizedSearchCV` (40 candidates × 5-fold CV) over XGBoost's key
hyperparameters. Tuned test performance: **RMSE 1208.75, MAE 571.11,
R² 0.9740** — an improvement over the untuned model. Saved as
`models/best_model.joblib`.

### Explainability (`06_explainability.py`)
- `plots/08_feature_importance.png` — top 15 features by model importance
- `plots/09_shap_summary.png` — SHAP beeswarm (global feature impact/direction)
- `plots/10_shap_waterfall_example.png` — SHAP waterfall for one prediction

Top drivers of price: car age, luxury-brand tier, kilometers driven, and
the model-level average price (target encoding).

## 5. API

`POST /predict` — see `api/app.py`. Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota", "model": "Toyota Corolla", "year": 2019,
    "km_driven": 45000, "fuel_type": "Petrol", "transmission": "Manual",
    "owner_type": "First Owner", "seller_type": "Individual",
    "mileage_kmpl": 18.5, "engine_cc": 1500, "max_power_bhp": 110,
    "seats": 5, "location": "Hyderabad"
  }'
# -> {"predicted_price_usd": 10868.22, "currency": "USD", "model_used": "XGBoost (tuned)"}
```

Interactive docs at `/docs` (Swagger UI) once the server is running.

## 6. Web UI

`web/index.html` is a single self-contained HTML/CSS/JS page (no build
step) with a form for all car attributes, an editable "API endpoint"
field, and a result card. Works by opening the file directly in a browser
or serving it statically.

## 7. Deployment

- `deployment/Dockerfile` — containerizes the FastAPI service
- `deployment/docker-compose.yml` — runs the API and a static file server
  for the web UI together
- `deployment/requirements.txt` — pinned Python dependencies

For a cloud deployment, push the built image to any container registry
and deploy it on services like AWS ECS/Fargate, Google Cloud Run, Azure
Container Apps, or Render/Railway — all of which can run the Dockerfile
as-is (expects `PORT`/`8000` and exposes `/health` for health checks).

## 8. Notes & next steps

- Swap in a real dataset (e.g. a Kaggle used-car dataset) by replacing
  `data/raw/car_data.csv` and re-running `run_pipeline.sh` from step 01.
- Add authentication / rate limiting to the API before public deployment.
- Consider model monitoring (data drift, prediction drift) once in
  production, and periodic retraining as new sales data comes in.
