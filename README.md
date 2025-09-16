# BFSI Predictive Modeling


## 🔎 Project Summary

The BFSI Predictive Modelling project is an AI-driven solution designed to strengthen the security and reliability of financial transactions. It leverages Large Language Models (LLMs) along with data analysis techniques to study historical customer transactions and behavioral patterns, and then predict future transactions while assessing fraud risks in real time.

> **Current state:** Data Collection, preprocessing , EDA. Modeling, Training and  Evaluation are the next steps.

  ## 📁 Folder Structure (relevant)

```
project_root/
├── data/                    # All datasets used in the project
│   ├── raw/                 # Original, unmodified data
│   ├── processed/           # Cleaned and processed data
├── notebooks/               # Jupyter notebooks for exploration, EDA, and experiments
├── src/                     # Source code of the project
│   ├── preprocessing/       # Scripts for data cleaning, feature engineering, transformations
│   ├── utils/               # Helper functions, common utilities
├── docs/                    # Project documentation, reports, and references
├── configs/                 # Configuration files (YAML/JSON) for training, experiments, etc.
├── tests/                   # Unit tests to ensure code correctness
├── requirements.txt         # List of Python dependencies
└── README.md                # Main project description and usage guide

```

## ⚙️ Technologies Used
pandas

numpy

matplotlib

seaborn

## ✅ What we have done

* Cleaned and prepared the transaction dataset for analysis.
* Feature Engineering: Extracting time-based features (hour, day, weekday) and creating a binary indicator for high-value transactions.
* Exploratory Data Analysis (EDA): Using matplotlib and seaborn to visualize patterns, trends, and anomalies in customer transactions.
* Data Transformation: Normalizing and structuring data for better model input.
* Model Preparation: Setting up the pipeline to integrate future machine learning and LLM-based predictive models.
