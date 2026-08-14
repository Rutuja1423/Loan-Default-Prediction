# Production Grade AI Loan Default Prediction & Credit Risk Analytics System

[![CI Pipeline](https://github.com/Rutuja1423/Loan-Default-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rutuja1423/Loan-Default-Prediction/actions)

## Author & Contributor

| Role | Name | GitHub Account |
|---|---|---|
| Project Owner & Author | Rutuja Shinde | [@Rutuja1423](https://github.com/Rutuja1423) |
| Sole Contributor | Rutuja Shinde | [@Rutuja1423](https://github.com/Rutuja1423) |

## Business Problem

Banks and financial institutions lose significant revenue when borrowers fail to repay loans. Approving every loan application increases default rates while rejecting too many applicants reduces profitability. This system addresses the core challenge: **How can we predict which loan applicants are likely to default and what risk factors drive those defaults?**

This project delivers a complete production grade credit risk analytics and machine learning pipeline that:
* Cleans and validates raw data with robust imputations and defensive checks
* Engineers financial risk signals before model input
* Trains and tunes candidate ML models using **model specific scikit-learn Pipelines** to prevent data leakage
* Evaluates models with 5 Fold Stratified Cross Validation and hyperparameter optimization via `RandomizedSearchCV`
* Calibrates default probabilities using `CalibratedClassifierCV` for reliable financial decision making
* Optimizes decision thresholds using **business cost minimization** (cost of missed default vs false alarm)
* Provides model explainability with **SHAP** (persisted for fast reload)
* Delivers a single or batch inference CLI (`predict.py`), Streamlit Web App, Power BI datasets and automated test suite

## Technologies Used

| Category | Technology |
|---|---|
| Programming | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Database | SQLite (Normalized Schema) |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Model Calibration | `CalibratedClassifierCV` (Isotonic regression) |
| Hyperparameter Optimization | `RandomizedSearchCV` (Stratified CV) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Code Quality & Linting | Ruff, Black, isort, Pre commit hooks |
| Testing | Pytest |
| CI/CD | GitHub Actions Workflow |
| Automation | Makefile (`make train`, `make test`, `make lint`) |
| Dashboard & UI | Power BI, Streamlit Web Application |

## Dataset

**Source:** Loan Default Prediction Dataset (255347 records, 18 raw features)

| Feature | Description |
|---|---|
| LoanID | Unique loan identifier |
| Age | Borrower age (invalid ages imputed via median) |
| Income | Annual income ($15000 to $149999) |
| LoanAmount | Loan amount requested |
| CreditScore | Credit score (300 to 850) |
| MonthsEmployed | Employment duration in months |
| NumCreditLines | Number of open credit lines |
| InterestRate | Loan interest rate (%) |
| LoanTerm | Loan term in months |
| DTIRatio | Debt to Income ratio (0.1 to 0.9) |
| Education | High School / Bachelor's / Master's / PhD |
| EmploymentType | Full time / Part time / Self employed / Unemployed |
| MaritalStatus | Single / Married / Divorced |
| HasMortgage | Yes / No |
| HasDependents | Yes / No |
| LoanPurpose | Auto / Business / Education / Home / Other |
| HasCoSigner | Yes / No |
| **Default** | **Target Variable: 0 (Non Default) / 1 (Default)** |

**Class Distribution:** 88.4% Non Default (225694) / 11.6% Default (29653)

## Project Structure

```
LoanDefaultPrediction/
* .github/
  * workflows/
    * ci.yml                     # GitHub Actions CI workflow
* app/
  * streamlit_app.py               # Interactive web application
* dashboard/                         # Power BI exports and setup guide
  * prepare_dashboard_data.py
* data/
  * raw/                           # Raw input dataset
  * cleaned/                       # Cleaned and engineered CSV datasets
* images/                            # Generated EDA & evaluation charts
* models/                            # Saved model pipelines (.pkl) and artifacts
* reports/                           # Performance reports and JSON evaluation metrics
* sql/                               # SQLite schema creation and analytical queries
* tests/                             # Automated Pytest test suite
  * test_data_cleaning.py
  * test_feature_engineering.py
  * test_model_training.py
* .pre-commit-config.yaml            # Pre commit hooks configuration
* config.py                          # Centralized configuration (Pathlib, hyperparams, paths)
* data_cleaning.py                   # Step 1: Cleaning & validation
* eda.py                             # Step 2: Exploratory Data Analysis
* feature_engineering.py             # Step 4: Domain feature creation
* model_training.py                  # Step 5: Model training, tuning, CV, calibration
* model_evaluation.py                # Step 6: Evaluation, SHAP, scorecard, business cost
* main.py                            # Full pipeline orchestrator
* predict.py                         # CLI for single & batch inference
* Makefile                           # Automation commands (train, test, lint, format)
* requirements.txt                   # Dependency specifications
* README.md                          # Project documentation
```

## Quickstart & Commands

### 1. Installation
```bash
git clone https://github.com/Rutuja1423/Loan-Default-Prediction.git
cd "Loan Default Prediction"
pip install -r requirements.txt
```

### 2. Using Makefile
```bash
make train      # Run full machine learning pipeline
make test       # Run pytest unit test suite
make lint       # Run ruff code style check
make format     # Format code using black, isort and ruff
```

### 3. Run Pipeline via CLI
```bash
python main.py              # Execute entire pipeline (Steps 1 to 7)
python main.py --step 1 4 5  # Run specific steps only
```

### 4. Running Inference CLI (`predict.py`)
```bash
# Single prediction (JSON input)
python predict.py --json '{"Age": 35, "Income": 65000, "LoanAmount": 25000, "CreditScore": 680, "MonthsEmployed": 48, "NumCreditLines": 4, "InterestRate": 10.5, "LoanTerm": 36, "DTIRatio": 0.35, "Education": "Bachelor's", "EmploymentType": "Full time", "MaritalStatus": "Married", "LoanPurpose": "Auto", "HasMortgage": "No", "HasDependents": "No", "HasCoSigner": "Yes"}'

# Batch prediction from CSV
python predict.py --csv input_batch.csv --output output_predictions.csv
```

### 5. Launch Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```

## Key Production Engineering Features

### 1. Robust Data Validation & Imputation
* Invalid ages outside [18, 100] are flagged as missing (`NaN`) and imputed using median values rather than clipped to avoid fabricating demographic data.
* Negative income and zero or negative loan amounts are safely dropped.
* Defensive column checks ensure missing keys fail early with informative errors.

### 2. Model Specific Preprocessing Pipelines
* **Linear Models (Logistic Regression):** Include `StandardScaler` and `drop_first=True` encoding inside an isolated scikit-learn `Pipeline` to prevent data leakage and multicollinearity.
* **Tree Based Models (Random Forest, XGBoost, LightGBM):** Retain unscaled features and full categorical indicator sets (`drop_first=False`) to preserve split interpretability.

### 3. Cross Validation & Probability Calibration
* Every model is evaluated using **5 Fold Stratified Cross Validation**.
* The winning model is wrapped with `CalibratedClassifierCV` (Isotonic regression) to output true well calibrated default probabilities.

### 4. Business Cost & Threshold Optimization
Instead of assuming a default 0.5 decision threshold, the evaluation module sweeps thresholds (0.01 to 0.99) to find the threshold minimizing **Expected Financial Loss**:

Expected Loss = (False Negatives x $10000) + (False Positives x $500)

### 5. SHAP Explainability & Persistence
* Computes Tree & Linear SHAP summary, bar and waterfall plots.
* SHAP values are persisted to disk (`models/shap_values.pkl`) to avoid costly recomputation during analysis runs.

### 6. Automated Testing & CI/CD
* Pytest suite covers data validation, ratio formulas, risk indicators and pipeline execution.
* GitHub Actions workflow runs linter checks and tests on Python 3.10, 3.11 and 3.12 for every push and pull request.

## License

This project is open source under the MIT License.
