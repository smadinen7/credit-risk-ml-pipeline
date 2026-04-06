# Credit Risk ML Pipeline

End-to-end machine learning pipeline for credit default prediction — from raw data to a containerized REST API with full experiment tracking.

---

## Overview

A production-style ML pipeline built around credit default prediction. The goal is not just model accuracy, but demonstrating the full lifecycle: data → features → experiments → model registry → API → container. Built to be reproducible, observable, and deployable.

---

## What This Project Does

**Data & Features**
- Data cleaning, missing value treatment, outlier handling
- Feature engineering: debt ratios, utilization rates, payment history signals
- Feature selection and importance analysis

**Modeling**
- XGBoost and LightGBM with hyperparameter tuning
- All experiments tracked with MLflow (parameters, metrics, artifacts)
- Model registered and versioned in the MLflow Model Registry

**Serving**
- FastAPI REST endpoint: `POST /predict` returns default probability + risk tier
- Input validation via Pydantic
- Containerized with Docker for reproducible deployment

**Explainability**
- SHAP values for global and local feature importance
- Per-prediction explanations surfaced through the API

**Dashboard**
- Streamlit app for interactive predictions and SHAP visualizations

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — real-world credit application data with 300K+ records.

---

## Stack

| Component | Tool |
|-----------|------|
| Modeling | XGBoost, LightGBM, Scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Containerization | Docker |
| Environment | Python 3.11 |

---

## Project Structure

```
credit-risk-ml-pipeline/
├── data/                   # Raw and processed datasets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_explainability.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py
├── app/
│   └── dashboard.py        # Streamlit dashboard
├── mlruns/                 # MLflow tracking directory
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
# Train and log to MLflow
python src/train.py

# Launch MLflow UI
mlflow ui

# Start the API
uvicorn api.main:app --reload

# Start the dashboard
streamlit run app/dashboard.py

# Or run everything with Docker
docker-compose up
```

---

## Key Results

*In progress — model metrics and SHAP plots will be added here.*
