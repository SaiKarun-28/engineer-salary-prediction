# Engineer Salary Prediction

Predicting software engineer annual salaries using Linear Regression with Lasso-based feature selection, VIF analysis, and log-transformed targets.

---

## Overview

This project tackles a real-world HR/Recruitment problem: estimating software engineer compensation based on measurable attributes like technical proficiency, education, company size, and geographic location. The analysis goes beyond simple model training — it walks through the full data science lifecycle including data cleaning, multicollinearity diagnostics, feature engineering, regularization-based feature selection, and honest model evaluation.

## Dataset

The dataset contains records of software engineers across major US tech hubs with the following attributes:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `years_experience` | Years of professional experience | Continuous |
| `education_level` | 1 = Bachelor's, 2 = Master's, 3 = PhD | Ordinal |
| `skill_python` | Python proficiency (0.0 – 1.0) | Continuous |
| `skill_sql` | SQL proficiency (0.0 – 1.0) | Continuous |
| `skill_aws` | AWS proficiency (0.0 – 1.0) | Continuous |
| `skill_cloud` | General cloud proficiency (0.0 – 1.0) | Continuous |
| `company_size` | 1 = Startup, 2 = Mid-size, 3 = Enterprise | Ordinal |
| `location` | US tech hub city | Categorical |
| `salary` | Annual salary in USD (Target) | Continuous |

## Methodology

### 1. Data Cleaning & Anomaly Removal

Duplicate employee IDs were identified and removed. Engineers with 35+ years of experience were filtered out as domain-level anomalies (e.g., 47 years of experience in Silicon Valley earning $73k — less than a fresh graduate).

### 2. Feature Engineering

All four individual skill scores (`skill_python`, `skill_sql`, `skill_aws`, `skill_cloud`) were merged into a single composite `tech_skill_score` to eliminate severe multicollinearity (VIF dropped from 192 to below 14).

### 3. Log-Transformation

A `log1p` transformation was applied to the target variable (`salary`) to stabilize variance across the salary range (addressing heteroscedasticity) and improve linear regression assumptions.

### 4. Multicollinearity Diagnosis (VIF)

Variance Inflation Factor was computed for all features. Features with VIF > 10 were addressed through compositing or removal to ensure stable, interpretable coefficients.

### 5. Model Comparison & Feature Selection

Three models were trained and compared on identical train/test splits:

| Model | Train R² | Test R² | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 0.973 | 0.872 | $4,821 | $3,968 |
| Ridge (L2) | 0.972 | 0.875 | $4,776 | $3,932 |
| **Lasso (L1)** | **0.961** | **0.895** | **$4,364** | **$3,335** |

Lasso achieved the best generalization (highest Test R², lowest error) and automatically selected 4 features out of 10 by driving the rest to zero.

### 6. Final Model

A Linear Regression model was retrained using only the 4 Lasso-selected features:

| Final Metric | Value |
| :--- | :--- |
| Train R² | 0.967 |
| Test R² | 0.867 |
| Test MAE | $3,870 |

## Key Findings

1. **Tech Skills Dominate:** The composite `tech_skill_score` is the single most powerful predictor of salary — its standardized coefficient is 5x larger than any other feature.
2. **New York Premium:** Being located in New York provides a statistically significant salary premium compared to the baseline city (Austin).
3. **Education & Company Size:** Both act as independent positive multipliers for compensation.
4. **Experience is Redundant:** Once technical skills are accounted for, raw years of experience adds no independent predictive value — Lasso drove its coefficient to exactly zero.

## Tech Stack

- **Python 3**
- **numpy** & **pandas** — Data manipulation
- **matplotlib** & **seaborn** — Visualization
- **scikit-learn** — StandardScaler, LinearRegression, RidgeCV, LassoCV, train_test_split, metrics

## Project Structure

```text
engineer-salary-prediction/
├── engineer_salary_prediction.csv
├── Salary_Prediction_Analysis.ipynb
└── README.md
```

## Limitations

- **Small Sample Size:** The dataset contains fewer than 100 observations after cleaning. Results are illustrative; a production model would require thousands of rows.
- **Synthetic Data:** The dataset was generated for demonstration purposes and may not reflect real-world salary distributions with full fidelity.
- **Unobserved Variables:** The model does not account for equity/stock options, sign-on bonuses, specific tech stacks (React vs. Angular), or negotiation dynamics — all of which significantly impact total compensation in practice.

## How to Run

**1. Clone this repository:**

```bash
git clone https://github.com/<your-username>/engineer-salary-prediction.git
```

**2. Install dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**3. Open the notebook:**

```bash
jupyter notebook Salary_Prediction_Analysis.ipynb
```

**4. Run All Cells** (Kernel → Restart & Run All).
