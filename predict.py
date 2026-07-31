"""
Prediction CLI
===============

Command-line interface for running inference with the trained model.

Usage:
    # Single prediction via JSON
    python predict.py --json '{"Age": 35, "Income": 65000, ...}'

    # Batch prediction from CSV
    python predict.py --csv input.csv --output predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config import (
    BINARY_COLS,
    CREDIT_SCORE_BINS,
    CREDIT_SCORE_LABELS,
    EDUCATION_MAP,
    MODELS_DIR,
    NOMINAL_COLS,
    ORDINAL_COL,
    RISK_BINS,
    RISK_LABELS,
    setup_logging,
)

logger = setup_logging("predict")


def _load_artifacts(
    models_dir: Path,
) -> tuple:
    """Load the trained model, feature names, and optimal thresholds.

    Args:
        models_dir: Directory containing model artifacts.

    Returns:
        Tuple of (model, feature_names, thresholds_info).

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    model_path = models_dir / "best_model.pkl"
    features_path = models_dir / "feature_names.pkl"
    name_path = models_dir / "best_model_name.txt"
    thresholds_path = models_dir / "optimal_thresholds.pkl"

    for p in [model_path, features_path, name_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {p}. Run the training pipeline first."
            )

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    model_name = name_path.read_text().strip()

    thresholds = None
    if thresholds_path.exists():
        thresholds = joblib.load(thresholds_path)

    logger.info("Loaded model: %s", model_name)
    return model, feature_names, thresholds


def _compute_derived_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute engineered features for a single input row.

    Mirrors the logic in ``feature_engineering.py``.

    Args:
        row: Dict with raw input features.

    Returns:
        Dict with added derived features.
    """
    income = row["Income"]
    loan_amt = row["LoanAmount"]
    interest = row["InterestRate"]
    term = row["LoanTerm"]
    age = row["Age"]
    months_emp = row["MonthsEmployed"]
    credit = row["CreditScore"]
    dti = row["DTIRatio"]

    row["LoanToIncomeRatio"] = loan_amt / income if income > 0 else 0
    row["MonthlyIncome"] = income / 12.0

    monthly_r = (interest / 100.0) / 12.0
    if monthly_r > 0:
        pf = (1 + monthly_r) ** term
        row["EMI"] = loan_amt * monthly_r * pf / (pf - 1)
    else:
        row["EMI"] = loan_amt / term if term > 0 else 0

    row["EMIToIncomeRatio"] = row["EMI"] / row["MonthlyIncome"] if row["MonthlyIncome"] > 0 else 0
    adult_months = max((age - 18) * 12, 1)
    row["EmploymentRatio"] = months_emp / adult_months

    for i, upper in enumerate(CREDIT_SCORE_BINS[1:]):
        if credit <= upper:
            row["CreditScoreBand"] = CREDIT_SCORE_LABELS[i]
            break

    row["CompositeRiskIndex"] = (dti * interest) / (credit / 100.0) if credit > 0 else 0
    row["HighRiskFlag"] = 1 if (dti > 0.6 and credit < 600) else 0
    row["TotalInterestCost"] = (row["EMI"] * term) - loan_amt

    return row


def _encode_row(row: Dict[str, Any], feature_names: list) -> pd.DataFrame:
    """Encode a single input row into a feature vector.

    Args:
        row: Dict with raw + derived features.
        feature_names: Expected feature column order.

    Returns:
        Single-row DataFrame matching the model's expected input.
    """
    encoded = dict(row)

    # Binary encoding
    for col in BINARY_COLS:
        if col in encoded:
            encoded[col] = 1 if encoded[col] == "Yes" else 0

    # Ordinal education
    if ORDINAL_COL in encoded:
        encoded[ORDINAL_COL] = EDUCATION_MAP.get(encoded[ORDINAL_COL], 1)

    # One-hot nominal (create all possible dummies)
    nominal_expansions = {
        "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        "MaritalStatus": ["Divorced", "Married", "Single"],
        "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    }
    for col, categories in nominal_expansions.items():
        val = encoded.pop(col, None)
        for cat in categories:
            encoded[f"{col}_{cat}"] = 1 if val == cat else 0

    df = pd.DataFrame([encoded])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction for a single applicant.

    Args:
        input_data: Dict with applicant features.

    Returns:
        Dict with prediction results.
    """
    model, feature_names, thresholds = _load_artifacts(MODELS_DIR)

    row = _compute_derived_features(dict(input_data))
    X = _encode_row(row, feature_names)

    prob = float(model.predict_proba(X)[0, 1])

    # Use business-cost threshold if available, else 0.5
    threshold = 0.5
    if thresholds and "best_cost_threshold" in thresholds:
        threshold = thresholds["best_cost_threshold"]

    prediction = int(prob >= threshold)

    # Risk tier
    for i, upper in enumerate(RISK_BINS[1:]):
        if prob <= upper:
            risk_tier = RISK_LABELS[i]
            break
    else:
        risk_tier = RISK_LABELS[-1]

    result = {
        "default_probability": round(prob, 4),
        "predicted_class": prediction,
        "threshold_used": threshold,
        "risk_tier": risk_tier,
    }
    return result


def predict_batch(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Run predictions for a batch of applicants from CSV.

    Args:
        csv_path: Path to input CSV file.
        output_path: Optional path to save output CSV.

    Returns:
        DataFrame with predictions appended.
    """
    model, feature_names, thresholds = _load_artifacts(MODELS_DIR)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows for batch prediction.", len(df))

    threshold = 0.5
    if thresholds and "best_cost_threshold" in thresholds:
        threshold = thresholds["best_cost_threshold"]

    results = []
    for _, row_data in df.iterrows():
        row = _compute_derived_features(row_data.to_dict())
        X = _encode_row(row, feature_names)
        prob = float(model.predict_proba(X)[0, 1])
        results.append({
            "Default_Probability": round(prob, 4),
            "Predicted_Default": int(prob >= threshold),
            "Threshold": threshold,
        })

    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info("Saved predictions to %s", output_path)

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction — Inference CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", type=str, help="JSON string with single applicant data")
    group.add_argument("--csv", type=str, help="Path to CSV for batch prediction")
    parser.add_argument("--output", type=str, help="Output CSV path (for batch mode)")

    args = parser.parse_args()

    if args.json:
        data = json.loads(args.json)
        result = predict_single(data)
        print(json.dumps(result, indent=2))
    elif args.csv:
        predict_batch(args.csv, args.output)
