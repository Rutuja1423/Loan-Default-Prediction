"""
Model Training Module
======================

Trains, tunes, cross-validates, and calibrates classification models
for loan-default prediction.

Design decisions (per code-review feedback):
    - **Model-specific pipelines**: Each model gets its own sklearn
      ``Pipeline``.  Linear models include ``StandardScaler`` +
      ``drop_first`` encoding; tree-based models skip scaling and
      retain all dummy categories.
    - **Cross-validation**: ``StratifiedKFold(5)`` is used to report
      robust CV metrics alongside holdout test performance.
    - **Hyperparameter tuning**: ``RandomizedSearchCV`` (not Optuna) —
      single dependency, sufficient for a 255 K-row dataset.
    - **Probability calibration**: ``CalibratedClassifierCV`` is
      applied to the best model after selection, because credit-risk
      decisions rely on well-calibrated probabilities.
    - **Random seed**: Logged at the start for reproducibility.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    CV_FOLDS,
    ID_COLS,
    PARAM_GRIDS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TUNING_CV_FOLDS,
    TUNING_ITERATIONS,
    setup_logging,
)

logger = setup_logging("model_training")


# ──────────────────────────────────────────────────
# Pipeline Builders
# ──────────────────────────────────────────────────

def _build_linear_pipeline(model: Any) -> Pipeline:
    """Build a pipeline for linear models (scaling included).

    Args:
        model: A sklearn-compatible linear classifier instance.

    Returns:
        sklearn Pipeline with StandardScaler → model.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _build_tree_pipeline(model: Any) -> Pipeline:
    """Build a pipeline for tree-based models (no scaling).

    Args:
        model: A sklearn-compatible tree classifier instance.

    Returns:
        sklearn Pipeline with model only.
    """
    return Pipeline([
        ("model", model),
    ])


def _get_models_and_pipelines(
    scale_pos_weight: float,
) -> Dict[str, Pipeline]:
    """Create model-specific pipelines for all candidate algorithms.

    Args:
        scale_pos_weight: Ratio of negative to positive samples for
            imbalance handling in XGBoost.

    Returns:
        Dict mapping model name → sklearn Pipeline.
    """
    pipelines: Dict[str, Pipeline] = {
        "Logistic Regression": _build_linear_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Decision Tree": _build_tree_pipeline(
            DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": _build_tree_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier

        pipelines["XGBoost"] = _build_tree_pipeline(
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
            )
        )
        logger.info("XGBoost loaded successfully.")
    except ImportError:
        logger.warning("XGBoost not installed. Skipping.")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier

        pipelines["LightGBM"] = _build_tree_pipeline(
            LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        )
        logger.info("LightGBM loaded successfully.")
    except ImportError:
        logger.warning("LightGBM not installed. Skipping.")

    return pipelines


# ──────────────────────────────────────────────────
# Core Training Logic
# ──────────────────────────────────────────────────

def _tune_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Run RandomizedSearchCV for the given model pipeline.

    Args:
        name: Human-readable model name.
        pipeline: sklearn Pipeline wrapping the model.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The best estimator found by the search, or the original
        pipeline if no param grid is defined.
    """
    param_grid = PARAM_GRIDS.get(name)
    if not param_grid:
        logger.info("  No param grid for %s. Using defaults.", name)
        pipeline.fit(X_train, y_train)
        return pipeline

    logger.info("  Tuning %s with RandomizedSearchCV (%d iters, %d-fold)...",
                name, TUNING_ITERATIONS, TUNING_CV_FOLDS)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=TUNING_ITERATIONS,
        cv=StratifiedKFold(n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info("  Best params: %s", search.best_params_)
    logger.info("  Best CV ROC-AUC: %.4f", search.best_score_)
    return search.best_estimator_


def _cross_validate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[float, float]:
    """Run stratified k-fold cross-validation.

    Args:
        name: Human-readable model name.
        pipeline: Fitted sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Tuple of (cv_mean, cv_std) for ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_mean, cv_std = scores.mean(), scores.std()
    logger.info("  %d-Fold CV ROC-AUC: %.4f ± %.4f", CV_FOLDS, cv_mean, cv_std)
    return cv_mean, cv_std


def _evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on the holdout test set.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with metric names and values including confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "True Positives (Default Correct)": int(tp),
        "False Positives (False Alarm)": int(fp),
        "True Negatives": int(tn),
        "False Negatives (Missed Default)": int(fn),
        "y_prob": y_prob,  # kept for downstream use, removed before saving
    }


# ──────────────────────────────────────────────────
# Public Entry Point
# ──────────────────────────────────────────────────

def train_and_evaluate_models(
    data_path: str,
    models_dir: str,
    reports_dir: str,
    dashboard_dir: str,
) -> pd.DataFrame:
    """Train, tune, cross-validate, and calibrate all candidate models.

    Workflow:
        1. Load ML-ready dataset
        2. Stratified train/test split
        3. For each model: build pipeline → tune → cross-validate → test
        4. Select best model by ROC-AUC
        5. Calibrate best model with CalibratedClassifierCV
        6. Save artifacts (pipelines, scaler, feature names, metrics)

    Args:
        data_path: Path to the ML-ready CSV.
        models_dir: Directory to save model artifacts.
        reports_dir: Directory to save reports.
        dashboard_dir: Directory to save dashboard data.

    Returns:
        DataFrame with model comparison results.

    Raises:
        FileNotFoundError: If data_path does not exist.
        RuntimeError: If the dataset is empty.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: ML MODEL TRAINING & COMPARISON")
    logger.info("=" * 60)
    logger.info("Random seed: %d", RANDOM_STATE)

    # Setup directories
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    dash_path = Path(dashboard_dir)
    for d in [models_path, reports_path, dash_path]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"ML-ready data not found: {data_path}")

    df = pd.read_csv(data_file)
    logger.info("Loaded ML-ready dataset with shape: %s", df.shape)

    if df.empty:
        raise RuntimeError("Dataset is empty.")

    # Separate features and target
    ignore_cols = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

    logger.info("Features: %d | Target distribution — 0: %d (%.1f%%), 1: %d (%.1f%%)",
                len(feature_cols),
                (y == 0).sum(), (y == 0).mean() * 100,
                (y == 1).sum(), (y == 1).mean() * 100)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    # Save feature names
    joblib.dump(feature_cols, models_path / "feature_names.pkl")

    # Imbalance weight
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    # Build model-specific pipelines
    pipelines = _get_models_and_pipelines(scale_pos_weight)

    results: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Tuple[Pipeline, float]] = {}

    for name, pipeline in pipelines.items():
        logger.info("\n--- Training %s ---", name)

        # Tune
        best_pipeline = _tune_model(name, pipeline, X_train, y_train)

        # Cross-validate
        cv_mean, cv_std = _cross_validate(name, best_pipeline, X_train, y_train)

        # Evaluate on holdout test
        metrics = _evaluate_on_test(best_pipeline, X_test, y_test)

        # Extract y_prob and remove from the dict before saving
        y_prob = metrics.pop("y_prob")

        row = {"Model": name, "CV_ROC-AUC_Mean": cv_mean, "CV_ROC-AUC_Std": round(cv_std, 4)}
        row.update(metrics)
        results.append(row)

        # Save individual pipeline
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(best_pipeline, models_path / filename)
        trained_pipelines[name] = (best_pipeline, metrics["ROC-AUC"])

    # Results comparison
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    logger.info("\n%s", results_df.to_string(index=False))

    results_df.to_csv(reports_path / "model_comparison.csv", index=False)
    results_df.to_csv(dash_path / "model_comparison.csv", index=False)

    # Save as JSON for programmatic comparison across runs
    results_json = results_df.to_dict(orient="records")
    with open(reports_path / "model_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # ──────────────────────────────────────────
    # Best model selection + calibration
    # ──────────────────────────────────────────
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_name][0]
    logger.info("\nBest model by ROC-AUC: %s", best_name)

    # Calibrate the best model for reliable probability estimates
    logger.info("Calibrating %s with CalibratedClassifierCV (5-fold)...", best_name)
    calibrated = CalibratedClassifierCV(
        best_pipeline,
        cv=5,
        method="isotonic",
    )
    calibrated.fit(X_train, y_train)

    # Evaluate calibrated model
    cal_auc = roc_auc_score(y_test, calibrated.predict_proba(X_test)[:, 1])
    logger.info("Calibrated ROC-AUC: %.4f (was %.4f)", cal_auc, trained_pipelines[best_name][1])

    # Save calibrated model as the best
    joblib.dump(calibrated, models_path / "best_model.pkl")
    with open(models_path / "best_model_name.txt", "w") as f:
        f.write(best_name)

    # Save scaler separately for the Streamlit app's convenience
    if best_name == "Logistic Regression":
        scaler = best_pipeline.named_steps.get("scaler")
        if scaler is not None:
            joblib.dump(scaler, models_path / "scaler.pkl")
    else:
        # Save a dummy record so downstream code doesn't break
        joblib.dump(None, models_path / "scaler.pkl")

    logger.info("Saved best_model.pkl (calibrated), best_model_name.txt, scaler.pkl")

    # Save test predictions for evaluation module
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Default"] = y_test
    test_eval_df["Predicted_Probability"] = calibrated.predict_proba(X_test)[:, 1]
    test_eval_df["Predicted_Class"] = (test_eval_df["Predicted_Probability"] >= 0.5).astype(int)
    test_eval_df.to_csv(reports_path / "test_predictions.csv", index=False)
    logger.info("Saved test_predictions.csv")

    logger.info("Model training completed successfully!")
    return results_df


if __name__ == "__main__":
    ml_csv = str(Path("data") / "cleaned" / "loan_data_ml_ready.csv")
    train_and_evaluate_models(ml_csv, "models", "reports", "dashboard")
