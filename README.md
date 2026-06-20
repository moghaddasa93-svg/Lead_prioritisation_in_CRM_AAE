# Lead Prioritisation in CRM Systems

**From Prediction to Decision**  
2nd Semester Project — Group AAE  
AAU Business School, Aalborg University

This repository contains the code, notebooks, model artifacts, and Streamlit prototype developed as part of the MSc Business Data Science programme at Aalborg University.

The project investigates how behavioural lead data can be transformed into segmentation insights, conversion predictions, lead prioritisation, and AI-assisted CRM recommendations that support marketing and sales decision-making.

## Live Streamlit Prototype

https://lead-prioritisation-in-crm-systems-group-aae.streamlit.app/

---

# Repository Structure

```text
.
├── artifacts/
│   ├── metrics/
│   │   └── train_stats.json
│   └── model/
│       └── xgboost_model.pkl
│
├── data/
│   └── Lead Scoring.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── RQ1_Clustering.ipynb
│   └── RQ2_+_RQ3.ipynb
│
├── output/
│   └── rq3_multi_agent_recommendations-2.csv
│
├── .streamlit/
│   └── secrets.toml.example
│
├── streamlit_app.py
├── monitor.py
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Google Gemini API
- Jupyter Notebooks

---

# Authors

**Group AAE**  
Emina Gracanin
Amalie Hougaard Lang
Ali Moghadas

MSc Business Data Science  
AAU Business School  
Aalborg University

---

# Disclaimer

This repository was developed for educational and research purposes as part of the MSc Business Data Science programme at Aalborg University.

The system should be considered a proof-of-concept CRM decision-support prototype and not a production-ready enterprise CRM solution.
