"""
Configuration Module for Loan Default Prediction Pipeline
==========================================================

Centralizes all project paths, column definitions, validation rules,
hyperparameter grids, and logging configuration. All other modules
import from this file rather than hardcoding values.

Uses ``pathlib.Path`` for cross-platform path handling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
RAW_DATA: Path = BASE_DIR / "data" / "raw" / "Loan_default.csv"
CLEANED_DIR: Path = BASE_DIR / "data" / "cleaned"
CLEANED_CSV: Path = CLEANED_DIR / "loan_data_cleaned.csv"
FEATURES_CSV: Path = CLEANED_DIR / "loan_data_features.csv"
ML_READY_CSV: Path = CLEANED_DIR / "loan_data_ml_ready.csv"
SQL_DIR: Path = BASE_DIR / "sql"
DB_PATH: Path = SQL_DIR / "loan_database.db"
MODELS_DIR: Path = BASE_DIR / "models"
IMAGES_DIR: Path = BASE_DIR / "images"
REPORTS_DIR: Path = BASE_DIR / "reports"
DASHBOARD_DIR: Path = BASE_DIR / "dashboard"

# ──────────────────────────────────────────────
# Column Definitions
# ──────────────────────────────────────────────
TARGET_COL: str = "Default"
ID_COLS: List[str] = ["LoanID", "CustomerID"]
BINARY_COLS: List[str] = ["HasMortgage", "HasDependents", "HasCoSigner"]
NOMINAL_COLS: List[str] = ["EmploymentType", "MaritalStatus", "LoanPurpose"]
ORDINAL_COL: str = "Education"
EDUCATION_MAP: Dict[str, int] = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
}

# Engineered feature names (created in feature_engineering.py)
ENGINEERED_FEATURES: List[str] = [
    "LoanToIncomeRatio",
    "MonthlyIncome",
    "EMI",
    "EMIToIncomeRatio",
    "EmploymentRatio",
    "CreditScoreBand",
    "CompositeRiskIndex",
    "HighRiskFlag",
    "TotalInterestCost",
]

# ──────────────────────────────────────────────
# Validation Rules
# ──────────────────────────────────────────────
AGE_MIN: int = 18
AGE_MAX: int = 100
INCOME_MIN: float = 0.0
LOAN_AMOUNT_MIN: float = 0.0

# ──────────────────────────────────────────────
# ML Configuration
# ──────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
TUNING_ITERATIONS: int = 5
TUNING_CV_FOLDS: int = 3

# Credit score binning
CREDIT_SCORE_BINS: List[int] = [0, 580, 670, 740, 800, 1000]
CREDIT_SCORE_LABELS: List[int] = [0, 1, 2, 3, 4]  # Poor → Excellent

# Risk scorecard thresholds
RISK_BINS: List[float] = [-0.01, 0.30, 0.60, 1.01]
RISK_LABELS: List[str] = ["Low Risk (<30%)", "Medium Risk (30-60%)", "High Risk (>60%)"]

# Business cost parameters for threshold optimization
COST_FALSE_NEGATIVE: float = 10_000.0  # Cost of missing a default
COST_FALSE_POSITIVE: float = 500.0     # Cost of a false alarm

# ──────────────────────────────────────────────
# Hyperparameter Search Spaces (RandomizedSearchCV)
# ──────────────────────────────────────────────
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },
    "Decision Tree": {
        "model__max_depth": [5, 8, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [8, 10, 12],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.3],
    },
    "XGBoost": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__num_leaves": [15, 31],
    },
}


# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: int = logging.INFO


def setup_logging(name: str = "LoanDefault") -> logging.Logger:
    """Configure and return a named logger with console output.

    Args:
        name: Logger name (typically the module name).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when called multiple times
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler()
        handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
