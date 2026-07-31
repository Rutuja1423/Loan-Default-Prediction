"""
Feature Engineering Module
===========================

Creates domain-specific financial features from the cleaned dataset.

This module only creates *deterministic* derived features (ratios,
bins, flags) that are safe to compute before the train/test split.
Encoding and scaling are intentionally **not** performed here — those
transformations live inside the model-specific sklearn ``Pipeline``
objects in ``model_training.py`` to prevent data leakage.

Engineered features:
    - LoanToIncomeRatio
    - MonthlyIncome
    - EMI (Equated Monthly Installment via amortization formula)
    - EMIToIncomeRatio
    - EmploymentRatio (proportion of adult life spent employed)
    - CreditScoreBand (ordinal: Poor → Excellent)
    - CompositeRiskIndex (DTI × InterestRate / CreditScore)
    - HighRiskFlag (high DTI + low credit score)
    - TotalInterestCost
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    BINARY_COLS,
    CREDIT_SCORE_BINS,
    CREDIT_SCORE_LABELS,
    EDUCATION_MAP,
    ENGINEERED_FEATURES,
    NOMINAL_COLS,
    ORDINAL_COL,
    setup_logging,
)

logger = setup_logging("feature_engineering")


def _create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add financial ratio features to the DataFrame.

    Args:
        df: Input DataFrame with raw financial columns.

    Returns:
        DataFrame with added ratio columns.
    """
    df["LoanToIncomeRatio"] = df["LoanAmount"] / df["Income"]
    df["MonthlyIncome"] = df["Income"] / 12.0

    # EMI = [P × R × (1+R)^N] / [(1+R)^N − 1]
    monthly_r = (df["InterestRate"] / 100.0) / 12.0
    n_months = df["LoanTerm"]
    pow_factor = (1 + monthly_r) ** n_months

    df["EMI"] = np.where(
        monthly_r > 0,
        df["LoanAmount"] * monthly_r * pow_factor / (pow_factor - 1),
        df["LoanAmount"] / n_months,
    )
    df["EMIToIncomeRatio"] = df["EMI"] / df["MonthlyIncome"]

    return df


def _create_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add employment-related features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with EmploymentRatio added.
    """
    adult_months = np.maximum((df["Age"] - 18) * 12, 1)
    df["EmploymentRatio"] = df["MonthsEmployed"] / adult_months
    return df


def _create_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit-risk indicator features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with CreditScoreBand, CompositeRiskIndex,
        HighRiskFlag, and TotalInterestCost added.
    """
    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"],
        bins=CREDIT_SCORE_BINS,
        labels=CREDIT_SCORE_LABELS,
    ).astype(int)

    df["CompositeRiskIndex"] = (
        (df["DTIRatio"] * df["InterestRate"]) / (df["CreditScore"] / 100.0)
    )

    df["HighRiskFlag"] = np.where(
        (df["DTIRatio"] > 0.6) & (df["CreditScore"] < 600), 1, 0
    )

    df["TotalInterestCost"] = (df["EMI"] * df["LoanTerm"]) - df["LoanAmount"]

    return df


def engineer_features(
    cleaned_csv_path: str, output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the feature engineering pipeline.

    Creates domain features, then saves both an unencoded version
    (for EDA / reporting) and an ML-ready encoded version.

    Note: The encoded version here retains all dummy categories
    (``drop_first=False``).  Model-specific handling of
    multicollinearity happens inside the training pipelines.

    Args:
        cleaned_csv_path: Path to the cleaned CSV.
        output_dir: Directory to save output CSVs.

    Returns:
        Tuple of (unencoded_features_df, ml_ready_df).

    Raises:
        FileNotFoundError: If cleaned_csv_path does not exist.
        RuntimeError: If the loaded DataFrame is empty.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    csv_path = Path(cleaned_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d cleaned records for feature engineering.", len(df))

    if df.empty:
        raise RuntimeError("Loaded dataset is empty.")

    fe_df = df.copy()

    # Build features
    fe_df = _create_financial_ratios(fe_df)
    fe_df = _create_employment_features(fe_df)
    fe_df = _create_risk_indicators(fe_df)

    new_count = fe_df.shape[1] - df.shape[1]
    logger.info("Created %d new engineered features: %s", new_count, ", ".join(ENGINEERED_FEATURES))

    # Save unencoded feature-engineered data
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fe_unencoded_path = out_path / "loan_data_features.csv"
    fe_df.to_csv(fe_unencoded_path, index=False)
    logger.info("Saved feature-engineered dataset to: %s", fe_unencoded_path)

    # Create ML-ready encoded dataset
    ml_df = fe_df.copy()

    # Map binary columns
    for col in BINARY_COLS:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].map({"Yes": 1, "No": 0})

    # Map ordinal education
    if ORDINAL_COL in ml_df.columns:
        ml_df[ORDINAL_COL] = ml_df[ORDINAL_COL].map(EDUCATION_MAP)

    # One-hot encode nominal columns (drop_first=False for tree models)
    existing_nominal = [c for c in NOMINAL_COLS if c in ml_df.columns]
    if existing_nominal:
        ml_df = pd.get_dummies(ml_df, columns=existing_nominal, drop_first=False)

    # Convert booleans to int
    bool_cols = ml_df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        ml_df[bool_cols] = ml_df[bool_cols].astype(int)

    ml_encoded_path = out_path / "loan_data_ml_ready.csv"
    ml_df.to_csv(ml_encoded_path, index=False)
    logger.info("Saved ML-ready dataset to: %s", ml_encoded_path)
    logger.info("Final ML dataset shape: %s", ml_df.shape)

    logger.info("Feature Engineering completed successfully!")
    return fe_df, ml_df


if __name__ == "__main__":
    clean_csv = str(Path("data") / "cleaned" / "loan_data_cleaned.csv")
    out_dir = str(Path("data") / "cleaned")
    engineer_features(clean_csv, out_dir)
