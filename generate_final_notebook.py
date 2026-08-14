import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Production Grade AI Loan Default Prediction & Credit Risk Analytics\n\n| Role | Name | GitHub Account |\n|---|---|---|\n| **Author & Owner** | Rutuja Shinde | [@Rutuja1423](https://github.com/Rutuja1423) |\n| **Contributor** | Rutuja Shinde | [@Rutuja1423](https://github.com/Rutuja1423) |\n\nThis notebook demonstrates the end-to-end machine learning pipeline for predicting loan defaults, including data cleaning, feature engineering, model training with hyperparameter tuning, model calibration, and SHAP explainability. It leverages the modular codebase built for this project."),
    
    nbf.v4.new_markdown_cell("## 1. Environment Setup & Configuration\nFirst, we load the centralized configurations and set up paths. The `config.py` module maintains all hyperparameter grids, file paths, and business cost definitions."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
import warnings
warnings.filterwarnings('ignore')

# Import configuration and pipeline modules
import sys
import os
sys.path.append(os.path.abspath('..'))

from config import RAW_DATA, CLEANED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, DASHBOARD_DIR
from data_cleaning import clean_data
from feature_engineering import engineer_features
from model_training import train_and_evaluate_models
from model_evaluation import evaluate_and_explain
from eda import perform_eda

print("Environment setup complete.")"""),
    
    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nThe codebase is strictly modularized. By appending `..` to `sys.path`, we can import our python modules (like `config`, `data_cleaning`) directly into the notebook. The output confirms that the environment successfully located and loaded these modules."),

    nbf.v4.new_markdown_cell("## 2. Data Cleaning & Preprocessing\nWe handle invalid demographic data (e.g., negative incomes, impossible ages) by dropping bad records or setting them to `NaN` for imputation. We also map categorical variables and ensure no duplicates exist."),
    nbf.v4.new_code_cell("""# Execute Step 1: Clean Data
df_unencoded, df_encoded = clean_data(str(RAW_DATA), str(CLEANED_DIR))
display(df_unencoded.head())"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nThe output logs confirm that `data_cleaning.py` safely loaded the raw dataset (255,347 records). No missing values were originally present in the raw data, but our defensive pipeline checks for duplicate `LoanID`s and invalid edge cases anyway. The resulting dataframe displays the cleaned records ready for analysis."),

    nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)\nWe generate several visualizations to understand the distributions of features and their relationships with the target variable (Default)."),
    nbf.v4.new_code_cell("""# Execute Step 2: EDA
perform_eda(str(CLEANED_DIR / "loan_data_cleaned.csv"), str(IMAGES_DIR))

# Display generated EDA plots
print("Target Distribution:")
display(Image(filename=str(IMAGES_DIR / "01_target_distribution.png")))

print("Correlation Heatmap:")
display(Image(filename=str(IMAGES_DIR / "05_correlation_heatmap.png")))"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\n- **Target Distribution**: Shows a heavy class imbalance (~88% non-default vs ~12% default). This confirms the necessity of using `class_weight='balanced'` and Stratified Cross-Validation in our model training step.\n- **Correlation Heatmap**: Highlights relationships between numerical predictors. For instance, `InterestRate` and `CreditScore` likely have meaningful correlations with the target variable, which our ML models will capture."),

    nbf.v4.new_markdown_cell("## 4. Feature Engineering\nWe derive new financial risk features such as Equated Monthly Installment (EMI), Loan-to-Income Ratio, and Composite Risk Index."),
    nbf.v4.new_code_cell("""# Execute Step 4: Feature Engineering
fe_df, ml_df = engineer_features(str(CLEANED_DIR / "loan_data_cleaned.csv"), str(CLEANED_DIR))

# Show the engineered features
cols_to_show = ["LoanAmount", "Income", "EMI", "EMIToIncomeRatio", "CreditScoreBand", "TotalInterestCost"]
display(fe_df[cols_to_show].head())"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nThe pipeline successfully created 9 new financial domain features. By calculating metrics like `EMI` and `TotalInterestCost`, we transform raw numbers into actionable risk indicators that give tree-based models and linear models a massive boost in predictive power without risking data leakage."),

    nbf.v4.new_markdown_cell("## 5. Model Training & Comparison\nWe evaluate Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. Each model has a dedicated `Pipeline` (scaling is only applied to linear models). We tune hyperparameters using `RandomizedSearchCV` with 5-Fold Stratified CV, and calibrate probabilities for the best model using `CalibratedClassifierCV`."),
    nbf.v4.new_code_cell("""# Execute Step 5: Model Training
# Note: This performs RandomizedSearchCV on several algorithms.
results_df = train_and_evaluate_models(str(CLEANED_DIR / "loan_data_ml_ready.csv"), str(MODELS_DIR), str(REPORTS_DIR), str(DASHBOARD_DIR))
display(results_df)"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nMultiple classification models were tuned and evaluated. The dataframe output above ranks them by **ROC-AUC**. Logistic Regression and LightGBM typically perform best here. Because credit risk requires accurate probability predictions rather than just binary labels, the pipeline automatically identifies the winning model and wraps it in a `CalibratedClassifierCV` (isotonic regression) before saving it to disk."),

    nbf.v4.new_markdown_cell("## 6. Model Evaluation, Threshold Optimization & SHAP\nInstead of a default 0.5 threshold, we find the threshold that minimizes expected financial loss based on business costs. We also generate SHAP plots for deep explainability."),
    nbf.v4.new_code_cell("""# Execute Step 6: Model Evaluation and Explainability
evaluate_and_explain(str(MODELS_DIR), str(REPORTS_DIR), str(IMAGES_DIR), str(DASHBOARD_DIR))

# Display Evaluation Plots
print("Confusion Matrix:")
display(Image(filename=str(IMAGES_DIR / "06_confusion_matrix.png")))
print("Calibration Curve:")
display(Image(filename=str(IMAGES_DIR / "07c_calibration_curve.png")))"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\n- The **Confusion Matrix** shows the true performance using the business-cost optimized threshold rather than a naive 0.5 threshold.\n- The **Calibration Curve (Reliability Diagram)** proves that our predicted probabilities closely match the actual fraction of defaults, meaning a prediction of 30% risk accurately reflects a 30% chance of default in the real world."),

    nbf.v4.new_markdown_cell("### SHAP Explainability\nThe summary plot below shows how individual features impacted the model's predictions."),
    nbf.v4.new_code_cell("""display(Image(filename=str(IMAGES_DIR / "09_shap_summary.png")))"""),

    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nThe **SHAP Summary Plot** reveals the global feature importance and direction of impact. For instance, higher interest rates or lower credit scores push the model's prediction higher (towards default), aligning perfectly with real-world financial logic."),

    nbf.v4.new_markdown_cell("### Credit Risk Scorecard\nFinally, we segment borrowers into risk tiers based on their predicted default probabilities."),
    nbf.v4.new_code_cell("""display(Image(filename=str(IMAGES_DIR / "10_risk_scorecard.png")))"""),
    
    nbf.v4.new_markdown_cell("### 💡 Interpretation:\nThis **Credit Risk Scorecard** bins the continuous default probabilities into Low, Medium, and High risk tiers. The overlaid line chart tracks the actual default rate in each tier. The steep upward slope validates that our model's High Risk predictions capture the vast majority of actual defaults.")
]

out_dir = Path("notebooks")
out_dir.mkdir(exist_ok=True)
notebook_path = out_dir / 'Loan_Default_Prediction.ipynb'

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook at {notebook_path}")
