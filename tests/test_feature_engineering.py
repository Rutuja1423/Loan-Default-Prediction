import pandas as pd
import pytest
from feature_engineering import _create_financial_ratios, _create_employment_features, _create_risk_indicators


def test_financial_ratios():
    df = pd.DataFrame({
        'Income': [120000.0],
        'LoanAmount': [30000.0],
        'InterestRate': [12.0],
        'LoanTerm': [12]
    })
    result = _create_financial_ratios(df)
    assert result['LoanToIncomeRatio'].iloc[0] == pytest.approx(0.25)
    assert result['MonthlyIncome'].iloc[0] == pytest.approx(10000.0)
    assert 'EMI' in result.columns
    assert 'EMIToIncomeRatio' in result.columns
    assert result['EMI'].iloc[0] > 0


def test_employment_features():
    df = pd.DataFrame({
        'Age': [38],
        'MonthsEmployed': [120]
    })
    result = _create_employment_features(df)
    # Adult months = (38 - 18) * 12 = 240
    # Ratio = 120 / 240 = 0.5
    assert result['EmploymentRatio'].iloc[0] == pytest.approx(0.5)


def test_risk_indicators():
    df = pd.DataFrame({
        'CreditScore': [550],
        'DTIRatio': [0.7],
        'InterestRate': [15.0],
        'EMI': [1000.0],
        'LoanTerm': [12],
        'LoanAmount': [10000.0]
    })
    result = _create_risk_indicators(df)
    assert result['CreditScoreBand'].iloc[0] == 0  # Poor
    assert result['HighRiskFlag'].iloc[0] == 1
    assert result['TotalInterestCost'].iloc[0] == pytest.approx(2000.0)
