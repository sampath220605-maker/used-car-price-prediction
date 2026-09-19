# Used Car Price Prediction API - production image
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost/shap (libgomp needed by xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to serve predictions
COPY api /app/api
COPY models /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
