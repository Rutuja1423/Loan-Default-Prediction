import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from data_cleaning import _remove_duplicates, _fix_invalid_rows, _impute_missing, _validate_columns


def test_validate_columns_success():
    df = pd.DataFrame({'Age': [25], 'Income': [50000], 'Default': [0]})
    _validate_columns(df, ['Age', 'Income'])


def test_validate_columns_missing():
    df = pd.DataFrame({'Age': [25]})
    with pytest.raises(ValueError):
        _validate_columns(df, ['Age', 'Income'])


def test_remove_duplicates():
    df = pd.DataFrame({
        'LoanID': ['L1', 'L1', 'L2'],
        'Age': [25, 25, 30]
    })
    cleaned = _remove_duplicates(df)
    assert len(cleaned) == 2
    assert list(cleaned['LoanID']) == ['L1', 'L2']


def test_fix_invalid_rows_age_nan():
    df = pd.DataFrame({
        'Age': [10, 25, 150],
        'Income': [50000, 60000, 70000],
        'LoanAmount': [10000, 20000, 15000]
    })
    cleaned = _fix_invalid_rows(df)
    assert len(cleaned) == 3
    assert np.isnan(cleaned.loc[0, 'Age'])
    assert cleaned.loc[1, 'Age'] == 25
    assert np.isnan(cleaned.loc[2, 'Age'])


def test_fix_invalid_rows_drop_negative_income_loan():
    df = pd.DataFrame({
        'Age': [25, 30, 35],
        'Income': [-1000, 60000, 70000],
        'LoanAmount': [10000, 0, 15000]
    })
    cleaned = _fix_invalid_rows(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Age'] == 35


def test_impute_missing():
    df = pd.DataFrame({
        'Age': [25.0, np.nan, 35.0],
        'Income': [50000.0, 60000.0, np.nan],
        'Education': ['Bachelor\'s', None, 'PhD']
    })
    imputed = _impute_missing(df)
    assert imputed['Age'].isnull().sum() == 0
    assert imputed['Age'].iloc[1] == 30.0  # Median of 25 and 35
    assert imputed['Education'].isnull().sum() == 0
