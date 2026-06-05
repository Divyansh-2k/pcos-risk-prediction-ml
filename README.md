# PCOS Risk Prediction System

A multi-modal machine learning system to predict the risk of Polycystic Ovary Syndrome (PCOS) using hormonal, ovarian, clinical, and lifestyle data, with interpretable outputs using SHAP.

---

## Overview

This project builds separate machine learning models for different medical feature groups and combines them into a unified risk score. The system also explains predictions using SHAP to provide transparency in decision-making.

---

## Features

- Multi-domain modeling:
  - Hormonal
  - Ovarian
  - Clinical
  - Lifestyle
- Model selection using:
  - XGBoost
  - CatBoost
  - RandomForest + AdaBoost Ensemble
- Risk scoring on a 0–100 scale
- Risk categorization:
  - Low Risk (<25)
  - Moderate Risk (25–60)
  - High Risk (>60)
- Model interpretability:
  - SHAP global feature importance
  - SHAP individual patient explanations
- Evaluation using ROC-AUC

---

## Project Structure

* `data/PCOS_data.csv` — Raw PCOS study dataset
* `outputs/risk_scores.csv` — Generated patient risk predictions
* `outputs/plots/` — Data distribution and model performance charts
* `src/__init__.py` — Package initializer
* `src/data_loader.py` — Script to handle data loading and preprocessing
* `src/model.py` — Machine learning model definition and training script
* `src/evaluation.py` — Validation metrics and model evaluation functions
* `src/interpretability.py` — Feature importance and SHAP/LIME explanation scripts
* `config.py` — Global parameters and hyperparameters
* `main.py` — Central script to run the entire end-to-end pipeline
* `requirements.txt` — Python dependencies needed to run the project
* `README.md` — Project documentation and setup guide
---

## Installation

Install required dependencies:

pip install -r requirements.txt

---

## Usage

Run the main pipeline:

python main.py

---

## Outputs

The system generates:

- Risk scores file:
  - outputs/risk_scores.csv

- Visualizations:
  - ROC curves for each feature group
  - SHAP summary plots (feature importance)
  - SHAP waterfall plots (individual explanations)

---

## Model Pipeline

1. Data cleaning and preprocessing
2. Feature grouping into:
   - Hormonal
   - Ovarian
   - Clinical
   - Lifestyle
3. Model training and selection using cross-validation
4. Risk score computation (probability scaled to 0–100)
5. Combined risk calculation using weighted aggregation
6. SHAP-based interpretability

---

## Key Insights

- Ovarian and lifestyle features show strong predictive power
- Hormonal features provide moderate signal
- Clinical features contribute additional contextual information
- SHAP reveals feature-level influence on predictions for each patient

---

## Notes

- This project is intended for academic and research purposes only

---

## Author

Divyansh Sharma
