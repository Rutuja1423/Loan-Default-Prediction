"""
AI-Based Loan Default Prediction and Credit Risk Analytics System
=================================================================
Main Pipeline Orchestrator

Runs the end-to-end pipeline with proper logging, error handling,
and step-level timing.

Usage:
    python main.py              # Run entire pipeline
    python main.py --step 1     # Run only data cleaning
    python main.py --step 2     # Run only EDA
    python main.py --step 3     # Run only SQL database & analysis
    python main.py --step 4     # Run only feature engineering
    python main.py --step 5     # Run only model training
    python main.py --step 6     # Run only model evaluation & SHAP
    python main.py --step 7     # Run only Power BI data export
    python main.py --step 1 2 5 # Run specific steps
"""

import argparse
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Callable

from config import (
    BASE_DIR,
    CLEANED_CSV,
    CLEANED_DIR,
    DASHBOARD_DIR,
    DB_PATH,
    FEATURES_CSV,
    IMAGES_DIR,
    ML_READY_CSV,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA,
    REPORTS_DIR,
    SQL_DIR,
    setup_logging,
)

logger = setup_logging("pipeline")


def step_1_clean() -> None:
    """Step 1: Data Cleaning & Preprocessing."""
    from data_cleaning import clean_data
    clean_data(str(RAW_DATA), str(CLEANED_DIR))


def step_2_eda() -> None:
    """Step 2: Exploratory Data Analysis."""
    from eda import perform_eda
    perform_eda(str(CLEANED_CSV), str(IMAGES_DIR))


def step_3_sql() -> None:
    """Step 3: SQL Database Creation & Analysis."""
    sys.path.insert(0, str(SQL_DIR))
    from create_database import build_database
    from sql_analysis import run_sql_analysis
    build_database(str(CLEANED_CSV), str(DB_PATH))
    run_sql_analysis(str(DB_PATH), str(DASHBOARD_DIR))


def step_4_features() -> None:
    """Step 4: Feature Engineering."""
    from feature_engineering import engineer_features
    engineer_features(str(CLEANED_CSV), str(CLEANED_DIR))


def step_5_train() -> None:
    """Step 5: ML Model Training & Comparison."""
    from model_training import train_and_evaluate_models
    train_and_evaluate_models(str(ML_READY_CSV), str(MODELS_DIR), str(REPORTS_DIR), str(DASHBOARD_DIR))


def step_6_evaluate() -> None:
    """Step 6: Model Evaluation, SHAP & Scorecard."""
    from model_evaluation import evaluate_and_explain
    evaluate_and_explain(str(MODELS_DIR), str(REPORTS_DIR), str(IMAGES_DIR), str(DASHBOARD_DIR))


def step_7_dashboard() -> None:
    """Step 7: Power BI Dashboard Data Export."""
    sys.path.insert(0, str(DASHBOARD_DIR))
    from prepare_dashboard_data import prepare_all_powerbi_data
    prepare_all_powerbi_data(str(CLEANED_CSV), str(FEATURES_CSV), str(DASHBOARD_DIR))


STEPS: Dict[int, Tuple[str, Callable]] = {
    1: ("Data Cleaning & Preprocessing", step_1_clean),
    2: ("Exploratory Data Analysis (EDA)", step_2_eda),
    3: ("SQL Database Creation & Analysis", step_3_sql),
    4: ("Feature Engineering", step_4_features),
    5: ("ML Model Training & Comparison", step_5_train),
    6: ("Model Evaluation, SHAP & Scorecard", step_6_evaluate),
    7: ("Power BI Dashboard Data Export", step_7_dashboard),
}


def run_pipeline(steps_to_run: Optional[List[int]] = None) -> None:
    """Execute the pipeline for the specified steps.

    Args:
        steps_to_run: List of step numbers to execute. If None, all
            steps are run in sequence.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  AI-BASED LOAN DEFAULT PREDICTION SYSTEM")
    logger.info("  Full Pipeline Execution")
    logger.info("  Random Seed: %d", RANDOM_STATE)
    logger.info("=" * 60)

    if steps_to_run is None:
        steps_to_run = list(STEPS.keys())

    total_start = time.time()
    failed_steps: List[int] = []

    for step_num in steps_to_run:
        if step_num not in STEPS:
            logger.error("Unknown step: %d. Valid steps: %s", step_num, list(STEPS.keys()))
            continue

        name, func = STEPS[step_num]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  PIPELINE STEP %d/%d: %s", step_num, len(STEPS), name)
        logger.info("=" * 60)

        step_start = time.time()
        try:
            func()
            elapsed = time.time() - step_start
            logger.info("  Step %d completed in %.1fs", step_num, elapsed)
        except FileNotFoundError as e:
            logger.error("  Step %d FAILED — file not found: %s", step_num, e)
            failed_steps.append(step_num)
        except Exception as e:
            logger.error("  Step %d FAILED: %s", step_num, e)
            logger.debug(traceback.format_exc())
            failed_steps.append(step_num)
            logger.info("  Continuing to next step...")

    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    if failed_steps:
        logger.warning("  PIPELINE COMPLETE WITH ERRORS | Failed steps: %s | Total Time: %.1fs",
                        failed_steps, total_elapsed)
    else:
        logger.info("  PIPELINE COMPLETE  |  Total Time: %.1fs", total_elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction Pipeline")
    parser.add_argument(
        "--step", type=int, nargs="+",
        help="Run specific step(s) only. E.g. --step 1 2 5",
    )
    args = parser.parse_args()
    run_pipeline(args.step)
