# Lead Prioritisation in CRM - Group AAE

This repository contains the code and project structure for the 2nd semester Business Data Science project at Aalborg University.

## Project Overview

The project focuses on a CRM-style lead prioritisation use case. The goal is to explore how data-driven methods can support sales and marketing teams in prioritising leads, estimating conversion likelihood, identifying lead segments, and generating actionable recommendations.

The system is designed as a proof-of-concept decision-support tool, not as a fully production-ready CRM platform.

## Main Research Question

How can data-driven methods support lead prioritisation and decision-making in CRM-based systems?

## Sub-Research Questions

1. Which lead characteristics and behavioural factors are most strongly associated with conversion likelihood?
2. Are there distinct patterns or segments among leads that influence prioritisation strategies?
3. How do different data-driven approaches affect lead prioritisation performance?
4. How can machine learning outputs be integrated into a CRM-style decision-support system?
5. How can system-generated insights support actionable recommendations for marketing and sales teams?

## Project Scope

The project builds on a previous MLOps lead scoring project and extends it toward a CRM-style decision-support system.

### Must-have components

- Data cleaning and preprocessing
- Feature selection and feature-to-RQ mapping
- Logistic Regression baseline
- Random Forest model
- Simple Neural Network / MLP
- Model comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- K-means clustering and PCA visualisation
- FastAPI prediction layer
- Streamlit CRM-style dashboard
- SQLite storage for leads, predictions, and feedback
- Rule-based priority labels and recommendations
- Basic logging and monitoring
- Ethics, GDPR, limitations, and critical reflection

### Nice-to-have components

- LLM-assisted recommendation generation
- SHAP or advanced explainability
- Hyperparameter tuning
- Cross-validation
- Advanced missing value imputation
- Bulk CSV upload
- Improved UI/UX

### Future work

- Real CRM integration
- Real-time data ingestion
- Automated retraining
- Alerting system
- Full CI/CD pipeline
- User authentication and role management
- Production-grade monitoring


## Repository Structure

The repository structure is intentionally kept simple at the beginning of the project and will be extended as the implementation develops. The structure reflects the project scope: data preprocessing, model comparison, clustering, CRM-style decision support, lightweight deployment, logging, monitoring, and reporting.

```text
Lead_prioritisation_in_CRM_AAE/
│
├── README.md
├── requirements.txt
├── .gitignore
├── docker-compose.yml              # Added/updated when Docker setup is ready
│
├── data/
│   ├── raw/                         # Original dataset
│   ├── processed/                   # Cleaned/preprocessed datasets
│   └── sample/                      # Small sample data for testing/demo
│
├── notebooks/
│   ├── 01_eda_and_feature_selection.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_clustering_and_pca.ipynb
│
├── src/
│   ├── preprocessing.py             # Data cleaning and preprocessing logic
│   ├── feature_engineering.py        # Feature creation and feature selection
│   ├── train_models.py              # Logistic Regression, Random Forest, MLP
│   ├── evaluate_models.py           # Model comparison and metrics
│   ├── clustering.py                # K-means clustering and PCA
│   ├── recommendations.py           # Rule-based priority labels and actions
│   ├── database.py                  # SQLite setup and database operations
│   └── monitoring.py                # Logging and basic monitoring metrics
│
├── api/
│   └── main.py                      # FastAPI prediction service
│
├── dashboard/
│   └── streamlit_app.py             # CRM-style Streamlit dashboard
│
├── artifacts/
│   ├── models/                      # Saved trained models
│   ├── preprocessors/               # Saved preprocessing pipeline
│   ├── clustering/                  # Saved clustering outputs/models
│   └── metrics/                     # Evaluation metrics and comparison results
│
├── reports/
│   ├── figures/                     # Plots for the report
│   └── screenshots/                 # Dashboard/API/system screenshots
│
└── configs/
    └── config.yaml                  # Optional configuration file

