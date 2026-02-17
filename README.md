# 🌫️ AQI Prediction System (End-to-End MLOps)

**Live App:**  
👉 https://aqi-prediction-4180.streamlit.app/

---

## 📌 Overview

Production-style end-to-end MLOps system for predicting Air Quality Index (AQI) using real-time weather and pollution data.

The system:

- Ingests hourly air quality + weather data
- Stores features in Hopsworks Feature Store
- Retrains models daily via GitHub Actions
- Registers best model in Hopsworks Model Registry
- Serves predictions via Streamlit
- Provides SHAP-based feature explanations

---

## 🏗️ Architecture

### Data Sources
- OpenWeather Weather API
- OpenWeather Air Pollution API

### Feature Store
- Hopsworks Cloud
- Feature Group: `aqi_features`
- ~180 days of hourly data (~4300+ rows)

### Training (Daily)
Models trained:
- Ridge Regression
- Random Forest
- Gradient Boosting
- SVR

Best model selected automatically based on lowest RMSE.

### Deployment
- Streamlit Cloud
- Loads latest model from registry
- Displays:
  - Predicted AQI
  - Actual vs Predicted (3-day trend)
  - AQI category bands
  - Latest feature snapshot

---

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|------|------|
| Ridge | 0.488 | 0.388 | 0.706 |
| Random Forest | 0.132 | 0.025 | 0.978 |
| Gradient Boosting | **0.131** | 0.064 | **0.979** |
| SVR | 0.429 | 0.349 | 0.773 |

**Selected Model:** Gradient Boosting

---

## 🔍 Explainability

SHAP analysis confirms:

- PM10 and PM2.5 are dominant predictors
- O3 significantly influences AQI
- Weather variables have secondary impact

SHAP plots are available in `/experiments`.

---

## 🔄 CI/CD Pipelines

| Workflow | Purpose |
|-----------|----------|
| `feature-hourly.yml` | Fetch + store hourly features |
| `train-daily.yml` | Retrain + register best model |
| `backfill-180.yml` | Historical data backfill |

All pipelines run via GitHub Actions.

---

## 📂 Project Structure

```text
aqi-prediction/
│
├── .gitignore                     # Excludes venv, cache, and sensitive files
│
├── requirements.txt               # Core dependencies for pipelines
│
├── feature_pipeline.py            # Hourly data ingestion + feature engineering
├── training_pipeline.py           # Daily model training + model registry update
│
├── eda.py                         # Exploratory Data Analysis script
├── EDA_Results.png                # AQI distribution visualization
├── EDA_Results2.png               # Correlation heatmap visualization
│
├── streamlit_app/                 # Deployment layer (UI)
│   ├── app.py                     # Streamlit dashboard application
│   ├── requirements.txt           # UI-specific dependencies
│   └── runtime.txt                # Python runtime version for deployment
│
├── experiments/                   # Offline experiments & model analysis
│   └── shap_explain.py            # SHAP feature importance analysis
│
└── .github/
    └── workflows/                 # CI/CD automation pipelines
        ├── feature-hourly.yml     # Runs hourly feature ingestion
        ├── train-daily.yml        # Runs daily model retraining
        └── backfill-180.yml       # Historical data backfill workflow
'''
---

## ▶️ Run Locally

Set environment variables:

-HOPSWORKS_HOST
-HOPSWORKS_PROJECT
-HOPSWORKS_API_KEY
-CITY


Run:

streamlit run streamlit_app/app.py

---

## 🚀 Key Highlights

- Fully automated retraining pipeline  
- Feature store–based reproducibility  
- Model registry integration  
- Real-time dashboard deployment  
- SHAP-based interpretability  
