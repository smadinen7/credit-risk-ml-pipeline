# Credit Risk ML Pipeline

End-to-end machine learning pipeline for credit default prediction — from raw data to a containerized REST API with full experiment tracking.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![MLflow](https://img.shields.io/badge/tracked%20with-MLflow-0194E2) ![FastAPI](https://img.shields.io/badge/served%20with-FastAPI-009688)

---

## Overview

A production-style ML pipeline built around credit default prediction. The goal is not just model accuracy, but demonstrating the full lifecycle: data → features → experiments → model registry → API → container. Built to be reproducible, observable, and deployable.

**Architecture:**
```
Raw Data (Kaggle)
    → Feature Engineering → MLflow Experiment Tracking
    → Model Registry (XGBoost / LightGBM)
    → FastAPI /predict endpoint
    → Docker container
    → Streamlit dashboard (SHAP visualizations)
```

---

## What This Project Does

**Data & Features**
- Data cleaning, missing value treatment, outlier handling
- Class imbalance: 8% positive rate handled via `scale_pos_weight` (XGBoost) and class weighting (LightGBM), with threshold tuning evaluated on precision-recall curves
- Feature engineering: debt ratios, utilization rates, payment history signals
- Feature selection via SHAP importance

**Modeling**
- XGBoost and LightGBM with Optuna hyperparameter tuning
- All experiments tracked with MLflow (parameters, metrics, artifacts)
- Best model registered and versioned in the MLflow Model Registry
- Evaluated on AUC-ROC, Gini coefficient, and F1 at optimized threshold

**Serving**
- FastAPI REST endpoint: `POST /predict` — returns default probability, risk tier, and top SHAP drivers
- Model loaded directly from MLflow registry: `mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")`
- Input validation via Pydantic
- Containerized with Docker; all services wired via docker-compose

**Explainability**
- SHAP TreeExplainer for global feature importance and per-prediction local explanations
- SHAP waterfall and summary plots embedded in dashboard and README

**Dashboard**
- Streamlit app with `@st.cache_resource` for SHAP computation
- Interactive single-prediction explorer and cohort-level analysis

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — `application_train.csv` only (307K rows, 122 features, 8% default rate).

**Setup (one-time):**
```bash
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json with chmod 600
kaggle competitions download -c home-credit-default-risk -f application_train.csv
unzip application_train.csv.zip -d data/raw/
```

---

## Stack

| Component | Tool |
|-----------|------|
| Modeling | XGBoost, LightGBM, Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Containerization | Docker, docker-compose |
| Environment | Python 3.11 |

**Version note:** Pin `xgboost>=2.0,<3`, `shap>=0.44`, `mlflow>=2.8`. SHAP TreeExplainer is sensitive to model library versions — train and serve with identical pinned versions.

---

## Quickstart

```bash
git clone https://github.com/smadinen7/credit-risk-ml-pipeline
cd credit-risk-ml-pipeline
# Download dataset (see Dataset section above)
pip install -r requirements.txt

# Train and log to MLflow
python src/train.py

# View experiments
mlflow ui   # localhost:5000

# Start the API
uvicorn api.main:app --reload   # localhost:8000

# Start the dashboard
streamlit run app/dashboard.py

# Or run everything with Docker
docker-compose up
```

**Sample API call:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amt_credit": 500000, "amt_income_total": 150000, "days_birth": -12000,
       "days_employed": -2000, "ext_source_2": 0.65, "ext_source_3": 0.55}'

# Response:
# {"default_probability": 0.12, "risk_tier": "Low", "top_shap_drivers": [...]}
```

---

## Project Structure

```
credit-risk-ml-pipeline/
├── data/
│   ├── raw/                    # Downloaded Kaggle data (gitignored)
│   └── processed/              # Engineered features (gitignored)
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
│   ├── main.py                 # FastAPI app — loads model from MLflow registry
│   └── schemas.py
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── mlruns/                     # MLflow tracking (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Key Results

*In progress — target metrics below. Results will be updated as experiments complete.*

| Model | AUC-ROC | Gini | F1 (threshold-tuned) |
|-------|---------|------|----------------------|
| XGBoost | — | — | — |
| LightGBM | — | — | — |

**Target:** AUC-ROC > 0.77 (baseline for this dataset in literature is ~0.75).

---

## Limitations

- Uses `application_train.csv` only; the full Home Credit dataset includes bureau history and previous application tables which would improve performance
- No concept drift detection or retraining pipeline — model is a point-in-time artifact
- Streamlit dashboard is for demo purposes only; not production-grade
