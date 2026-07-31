"""
Data Cleaning & Preprocessing Module
=====================================

Cleans the raw loan dataset by:
- Removing duplicates
- Setting invalid rows to NaN and imputing (not clipping)
- Imputing missing values (median for numeric, mode for categorical)
- Encoding categorical variables for ML consumption

Design decisions:
- Invalid ages are set to NaN and imputed (not clipped — clipping
  demographic attributes would fabricate data)
- Rows with negative income or non-positive loan amounts are dropped
  (these represent data-entry errors, not edge cases)
- ``drop_first=False`` is used here; model-specific pipelines in
  ``model_training.py`` handle multicollinearity per-model
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from config import (
    AGE_MAX,
    AGE_MIN,
    BINARY_COLS,
    EDUCATION_MAP,
    INCOME_MIN,
    LOAN_AMOUNT_MIN,
    NOMINAL_COLS,
    ORDINAL_COL,
    TARGET_COL,
    setup_logging,
)

logger = setup_logging("data_cleaning")


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing from the DataFrame.

    Args:
        df: Input DataFrame to check.
        required: List of column names that must be present.

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on LoanID if the column exists.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    if "LoanID" in df.columns:
        duplicate_count = df.duplicated(subset=["LoanID"]).sum()
        if duplicate_count > 0:
            df = df.drop_duplicates(subset=["LoanID"])
            logger.info("Removed %d duplicate LoanID rows. New shape: %s", duplicate_count, df.shape)
        else:
            logger.info("No duplicate LoanIDs found.")
    else:
        # Fallback: check full-row duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates()
            logger.info("LoanID column not found. Removed %d full-row duplicates.", duplicate_count)
        else:
            logger.info("LoanID column not found. No full-row duplicates detected.")
    return df


def _fix_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fix or remove rows with invalid values.

    Strategy:
    - Age outside [AGE_MIN, AGE_MAX]: set to NaN (will be imputed later)
    - Income < 0: drop rows
    - LoanAmount <= 0: drop rows

    Invalid ages are set to NaN rather than clipped because clipping
    demographic attributes fabricates data.  The subsequent imputation
    step will replace these NaNs with the column median.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with invalid rows handled.
    """
    initial_len = len(df)

    # Set invalid ages to NaN for imputation
    if "Age" in df.columns:
        invalid_age_mask = (df["Age"] < AGE_MIN) | (df["Age"] > AGE_MAX)
        invalid_age = invalid_age_mask.sum()
        if invalid_age > 0:
            df.loc[invalid_age_mask, "Age"] = np.nan
            logger.warning(
                "Set %d rows with Age outside [%d, %d] to NaN for imputation.",
                invalid_age, AGE_MIN, AGE_MAX,
            )

    # Drop negative income
    if "Income" in df.columns:
        invalid_income = (df["Income"] < INCOME_MIN).sum()
        if invalid_income > 0:
            df = df[df["Income"] >= INCOME_MIN]
            logger.warning("Dropped %d rows with Income < %.0f.", invalid_income, INCOME_MIN)

    # Drop non-positive loan amount
    if "LoanAmount" in df.columns:
        invalid_loan = (df["LoanAmount"] <= LOAN_AMOUNT_MIN).sum()
        if invalid_loan > 0:
            df = df[df["LoanAmount"] > LOAN_AMOUNT_MIN]
            logger.warning("Dropped %d rows with LoanAmount <= %.0f.", invalid_loan, LOAN_AMOUNT_MIN)

    removed = initial_len - len(df)
    if removed > 0:
        logger.info("Total rows dropped during validation: %d. Remaining: %d", removed, len(df))
    else:
        logger.info("No rows dropped during validation.")

    return df.reset_index(drop=True)


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, most-frequent for categorical.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values filled.
    """
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        logger.info("No missing values found. Skipping imputation.")
        return df

    logger.info("Found %d total missing values. Imputing...", missing_total)

    # Log per-column counts
    missing_per_col = df.isnull().sum()
    for col, count in missing_per_col[missing_per_col > 0].items():
        logger.info("  %s: %d missing", col, count)

    # Numeric imputation (median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Categorical imputation (most frequent via pandas mode for robust null/None handling)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode().dropna()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    logger.info("Missing value imputation complete.")
    return df


def clean_data(raw_filepath: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full data cleaning pipeline.

    Steps:
        1. Load raw CSV
        2. Validate required columns exist
        3. Remove duplicates
        4. Set invalid values to NaN / drop bad rows
        5. Impute missing values (median for numeric, mode for categorical)
        6. Save unencoded clean dataset
        7. Encode categorical features for ML
        8. Save encoded dataset

    Args:
        raw_filepath: Path to the raw CSV file.
        output_dir: Directory to save cleaned outputs.

    Returns:
        Tuple of (unencoded_df, encoded_df).

    Raises:
        FileNotFoundError: If raw_filepath does not exist.
        ValueError: If required columns are missing.
        RuntimeError: If DataFrame is empty after cleaning.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA CLEANING & PREPROCESSING")
    logger.info("=" * 60)

    # 1. Load dataset
    raw_path = Path(raw_filepath)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_filepath}")

    logger.info("Loading raw dataset from %s...", raw_filepath)
    df = pd.read_csv(raw_path)
    logger.info("Raw dataset shape: %s", df.shape)

    if df.empty:
        raise RuntimeError("Loaded dataset is empty.")

    # 2. Validate essential columns
    essential_cols = [TARGET_COL, "Age", "Income", "LoanAmount"]
    _validate_columns(df, essential_cols)

    # 3. Remove duplicates
    df = _remove_duplicates(df)

    # 4. Fix invalid rows (NaN for invalid ages, drop bad income/loan)
    df = _fix_invalid_rows(df)

    if df.empty:
        raise RuntimeError("DataFrame is empty after removing invalid rows.")

    # 5. Impute missing values
    df = _impute_missing(df)

    # 6. Save unencoded clean data (for EDA & SQL)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_csv_path = output_path / "loan_data_cleaned.csv"
    df.to_csv(clean_csv_path, index=False)
    logger.info("Saved unencoded cleaned data to: %s", clean_csv_path)

    # 7. Encode for ML
    # NOTE: drop_first=False here. Model-specific pipelines in
    # model_training.py use drop_first=True only for linear models.
    df_encoded = df.copy()

    # Map binary columns
    for col in BINARY_COLS:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({"Yes": 1, "No": 0})

    # Map ordinal education
    if ORDINAL_COL in df_encoded.columns:
        df_encoded["Education_Encoded"] = df_encoded[ORDINAL_COL].map(EDUCATION_MAP)

    # One-hot encode nominal categories (drop_first=False — trees use all)
    existing_nominal = [c for c in NOMINAL_COLS if c in df_encoded.columns]
    if existing_nominal:
        df_encoded = pd.get_dummies(df_encoded, columns=existing_nominal, drop_first=False)

    # Convert any bool columns to int
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    encoded_csv_path = output_path / "loan_data_encoded.csv"
    df_encoded.to_csv(encoded_csv_path, index=False)
    logger.info("Saved encoded cleaned data to: %s", encoded_csv_path)

    logger.info("Data cleaning & preprocessing completed successfully!")
    return df, df_encoded


if __name__ == "__main__":
    raw_path = Path("data") / "raw" / "Loan_default.csv"
    out_dir = Path("data") / "cleaned"
    clean_data(str(raw_path), str(out_dir))
