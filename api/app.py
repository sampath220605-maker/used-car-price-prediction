"""
FastAPI service for the Used Car Price Prediction model.

Run locally:
    cd used_car_price_prediction
    pip install -r deployment/requirements.txt
    uvicorn api.app:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI, or
open web/index.html in a browser (it calls this API at localhost:8000).
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predicts the resale price (USD) of a used car from its "
                 "specifications.",
    version="1.0.0",
)

# Allow the static web UI (served from file:// or any localhost port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model + preprocessing pipeline once at startup ----
def load_model_with_fallback():
    primary_path = os.path.join(MODELS_DIR, "best_model.joblib")
    candidate_paths = [primary_path]

    model_name_file = os.path.join(MODELS_DIR, "best_model_name.txt")
    if os.path.exists(model_name_file):
        try:
            with open(model_name_file, "r", encoding="utf-8") as f:
                model_name = f.read().strip()
            if model_name:
                candidate_paths.append(
                    os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
                )
        except OSError:
            pass

    last_error = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            return model, os.path.basename(path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to load a valid model from {candidate_paths}. Last error: {last_error}"
    )


model, model_filename = load_model_with_fallback()
pipeline = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))

scaler = pipeline["scaler"]
numeric_cols = pipeline["numeric_cols"]
feature_columns = pipeline["feature_columns"]
model_price_map = pipeline["model_price_map"]
global_mean = pipeline["global_mean"]
brand_tier_map = pipeline["brand_tier_map"]
owner_rank_map = pipeline["owner_rank"]
cat_cols = pipeline["cat_cols"]
MODEL_DISPLAY_NAME = "XGBoost" if "xgboost" in model_filename.lower() else "Model"

CURRENT_YEAR = 2026


class CarInput(BaseModel):
    brand: str = Field(..., example="Toyota")
    model_name: str = Field(..., alias="model", example="Toyota Corolla")
    year: int = Field(..., ge=1990, le=2026, example=2019)
    km_driven: float = Field(..., ge=0, example=45000)
    fuel_type: str = Field(..., example="Petrol")
    transmission: str = Field(..., example="Manual")
    owner_type: str = Field(..., example="First Owner")
    seller_type: str = Field(..., example="Individual")
    mileage_kmpl: float = Field(..., ge=0, example=18.5)
    engine_cc: float = Field(..., ge=0, example=1500)
    max_power_bhp: float = Field(..., ge=0, example=110)
    seats: int = Field(..., ge=2, le=9, example=5)
    location: str = Field(..., example="Hyderabad")

    class Config:
        populate_by_name = True


class PredictionOutput(BaseModel):
    predicted_price_usd: float
    currency: str = "USD"
    model_used: str


def build_feature_row(car: CarInput) -> pd.DataFrame:
    car_age = max(CURRENT_YEAR - car.year, 0)
    km_per_year = car.km_driven / max(car_age, 1)
    power_per_cc = car.max_power_bhp / car.engine_cc if car.engine_cc else 0.0
    brand_title = car.brand.strip().title()
    brand_tier = brand_tier_map.get(brand_title, "Mid")
    owner_rank = owner_rank_map.get(car.owner_type.strip().title(), 1)
    model_target_enc = model_price_map.get(car.model_name.strip().title(), global_mean)

    row = {
        "brand": brand_title,
        "km_driven": car.km_driven,
        "fuel_type": car.fuel_type.strip().title(),
        "transmission": car.transmission.strip().title(),
        "owner_type": car.owner_type.strip().title(),
        "seller_type": car.seller_type.strip().title(),
        "mileage_kmpl": car.mileage_kmpl,
        "engine_cc": car.engine_cc,
        "max_power_bhp": car.max_power_bhp,
        "seats": car.seats,
        "location": car.location.strip().title(),
        "car_age": car_age,
        "km_per_year": km_per_year,
        "power_per_cc": power_per_cc,
        "brand_tier": brand_tier,
        "owner_rank": owner_rank,
        "model_target_enc": model_target_enc,
    }
    df = pd.DataFrame([row])

    df_enc = pd.get_dummies(df, columns=cat_cols)
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)

    df_enc[numeric_cols] = scaler.transform(df_enc[numeric_cols])
    return df_enc[feature_columns]


@app.get("/")
def root():
    return {"status": "ok", "message": "Used Car Price Prediction API is running.",
            "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(car: CarInput):
    try:
        X = build_feature_row(car)
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)
        return PredictionOutput(
            predicted_price_usd=round(pred, 2),
            model_used=f"{MODEL_DISPLAY_NAME} (tuned)",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
