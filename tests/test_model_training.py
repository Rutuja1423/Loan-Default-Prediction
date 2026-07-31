import pandas as pd
import numpy as np
import pytest
from model_training import _build_linear_pipeline, _build_tree_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def test_build_linear_pipeline():
    model = LogisticRegression()
    pipeline = _build_linear_pipeline(model)
    assert 'scaler' in pipeline.named_steps
    assert 'model' in pipeline.named_steps


def test_build_tree_pipeline():
    model = RandomForestClassifier()
    pipeline = _build_tree_pipeline(model)
    assert 'scaler' not in pipeline.named_steps
    assert 'model' in pipeline.named_steps


def test_pipeline_fit_predict():
    X = pd.DataFrame({
        'Feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'Feature2': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    pipeline = _build_linear_pipeline(LogisticRegression())
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 10
    probs = pipeline.predict_proba(X)
    assert probs.shape == (10, 2)
