"""
Model Evaluation & Explainability Module
==========================================

Performs detailed evaluation of the best trained model including:
    - Confusion matrix visualization
    - Feature importance (tree or coefficient based)
    - SHAP explainability (summary, bar, and waterfall plots)
    - Precision-Recall curve + Average Precision
    - ROC curve
    - Calibration curve (reliability diagram)
    - KS Statistic (Kolmogorov-Smirnov — standard in credit risk)
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - Threshold optimization:
        a) Best F1 threshold
        b) Business-cost minimization threshold
    - Credit Risk Scorecard segmentation

SHAP values are persisted to disk to avoid recomputation on
subsequent runs.  All numeric metrics are saved as JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    RISK_BINS,
    RISK_LABELS,
    setup_logging,
)

logger = setup_logging("model_evaluation")


# ──────────────────────────────────────────────
# Threshold Optimization
# ──────────────────────────────────────────────

def _optimize_threshold_f1(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> Tuple[float, float]:
    """Find the threshold that maximizes F1-Score.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return round(best_t, 2), round(best_f1, 4)


def _optimize_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> Tuple[float, float]:
    """Find the threshold that minimizes expected business cost.

    Expected Loss = FN × cost_fn + FP × cost_fp

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        cost_fn: Dollar cost per missed default (false negative).
        cost_fp: Dollar cost per false alarm (false positive).

    Returns:
        Tuple of (best_threshold, minimum_expected_cost).
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return round(best_t, 2), round(best_cost, 2)


# ──────────────────────────────────────────────
# KS Statistic
# ──────────────────────────────────────────────

def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic.

    KS = max |CDF_positive − CDF_negative|

    Widely used in credit scoring to measure the separation between
    the distributions of defaulters and non-defaulters.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        KS statistic value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return round(float(ks), 4)


# ──────────────────────────────────────────────
# Plot Helpers
# ──────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Non-Default", "Default"],
        yticklabels=["Non-Default", "Default"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Actual Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


def _plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path,
) -> None:
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


def _plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path,
) -> None:
    """Save a Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#d62728", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


def _plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path,
) -> None:
    """Save a calibration (reliability) diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#2ca02c", lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


def _plot_ks_chart(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path,
) -> None:
    """Save a KS (Kolmogorov-Smirnov) chart."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    ks_max_idx = np.argmax(ks_values)
    ks_value = ks_values[ks_max_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, tpr[:-1] if len(tpr) > len(thresholds) else tpr,
            label="TPR (Sensitivity)", color="#1f77b4", lw=2)
    ax.plot(thresholds, fpr[:-1] if len(fpr) > len(thresholds) else fpr,
            label="FPR (1 − Specificity)", color="#d62728", lw=2)

    # Use the threshold at KS max for annotation
    if ks_max_idx < len(thresholds):
        ks_threshold = thresholds[ks_max_idx]
        ax.axvline(x=ks_threshold, color="gray", linestyle="--", alpha=0.7,
                   label=f"KS = {ks_value:.4f} @ {ks_threshold:.2f}")

    ax.set_xlabel("Threshold", fontweight="bold")
    ax.set_ylabel("Rate", fontweight="bold")
    ax.set_title(f"KS Chart — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


def _plot_risk_scorecard(
    risk_summary: pd.DataFrame, save_path: Path,
) -> None:
    """Save the risk-scorecard bar + line chart."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_bar = "#1f77b4"
    ax1.set_xlabel("Risk Category", fontweight="bold")
    ax1.set_ylabel("Applicant Count", color=color_bar, fontweight="bold")
    ax1.bar(
        risk_summary["RiskCategory"],
        risk_summary["Total_Applicants"],
        color=color_bar, alpha=0.7, width=0.5,
    )
    ax1.tick_params(axis="y", labelcolor=color_bar)

    ax2 = ax1.twinx()
    color_line = "#d62728"
    ax2.set_ylabel("Actual Default Rate (%)", color=color_line, fontweight="bold")
    ax2.plot(
        risk_summary["RiskCategory"],
        risk_summary["Actual_Default_Rate"],
        color=color_line, marker="o", linewidth=2.5, markersize=8,
    )
    ax2.tick_params(axis="y", labelcolor=color_line)

    plt.title(
        "Credit Risk Scorecard: Volume vs Default Rate",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path.name)


# ──────────────────────────────────────────────
# SHAP Analysis
# ──────────────────────────────────────────────

def _run_shap_analysis(
    best_model: Any,
    best_model_name: str,
    feature_names: list,
    pred_df: pd.DataFrame,
    images_dir: Path,
    models_dir: Path,
) -> None:
    """Compute SHAP values and generate summary, bar, and waterfall plots.

    SHAP values are saved to ``models_dir / shap_values.pkl`` so they
    don't need to be recomputed on every run.

    Args:
        best_model: The trained model (may be a Pipeline or calibrated wrapper).
        best_model_name: Human-readable model name.
        feature_names: List of feature column names.
        pred_df: DataFrame with test predictions.
        images_dir: Directory to save SHAP plots.
        models_dir: Directory to persist SHAP values.
    """
    import shap

    logger.info("Computing SHAP values for model explainability...")
    X_sample = pred_df[feature_names].sample(
        min(1000, len(pred_df)), random_state=42,
    )

    # Extract the underlying model from calibrated/pipeline wrappers
    underlying = best_model
    if hasattr(underlying, "estimator"):  # CalibratedClassifierCV
        underlying = underlying.estimator
    if hasattr(underlying, "named_steps"):  # Pipeline
        underlying = underlying.named_steps.get("model", underlying)

    # Choose appropriate explainer
    tree_keywords = ("Tree", "Forest", "XGB", "LGB", "Gradient")
    if any(kw in best_model_name for kw in tree_keywords) or hasattr(underlying, "feature_importances_"):
        explainer = shap.TreeExplainer(underlying)
    else:
        explainer = shap.LinearExplainer(underlying, X_sample)

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Persist SHAP values to avoid recomputation
    joblib.dump(
        {"shap_values": shap_values, "X_sample": X_sample},
        models_dir / "shap_values.pkl",
    )
    logger.info("Saved SHAP values to %s", models_dir / "shap_values.pkl")

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(images_dir / "09_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved 09_shap_summary.png")

    # Bar plot (global importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance ({best_model_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(images_dir / "09b_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved 09b_shap_bar.png")

    # Waterfall plot for a single prediction
    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if not isinstance(
                explainer.expected_value, np.ndarray
            ) else explainer.expected_value[1],
            data=X_sample.iloc[0].values,
            feature_names=list(X_sample.columns),
        )
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title("SHAP Waterfall — Single Prediction", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(images_dir / "09c_shap_waterfall.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved 09c_shap_waterfall.png")
    except Exception as e:
        logger.warning("Waterfall plot skipped: %s", e)


# ──────────────────────────────────────────────
# Public Entry Point
# ──────────────────────────────────────────────

def evaluate_and_explain(
    models_dir: str,
    reports_dir: str,
    images_dir: str,
    dashboard_dir: str,
) -> None:
    """Run full evaluation, explainability, and risk scorecard.

    Args:
        models_dir: Directory containing saved model artifacts.
        reports_dir: Directory containing test_predictions.csv.
        images_dir: Directory to save visualization PNGs.
        dashboard_dir: Directory to save dashboard CSVs.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: MODEL EVALUATION & EXPLAINABILITY")
    logger.info("=" * 60)

    m_dir = Path(models_dir)
    r_dir = Path(reports_dir)
    i_dir = Path(images_dir)
    d_dir = Path(dashboard_dir)
    for d in [i_dir, d_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    best_model_path = m_dir / "best_model.pkl"
    feature_names_path = m_dir / "feature_names.pkl"
    model_name_path = m_dir / "best_model_name.txt"
    predictions_path = r_dir / "test_predictions.csv"

    for path in [best_model_path, feature_names_path, model_name_path, predictions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    best_model = joblib.load(best_model_path)
    feature_names = joblib.load(feature_names_path)
    best_model_name = model_name_path.read_text().strip()

    logger.info("Evaluating best model: %s", best_model_name)

    # Load predictions
    pred_df = pd.read_csv(predictions_path)
    y_test = pred_df["Actual_Default"].values
    y_prob = pred_df["Predicted_Probability"].values
    y_pred = pred_df["Predicted_Class"].values

    # ─── Core Metrics ───
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    ks = _ks_statistic(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    logger.info("ROC-AUC: %.4f | Avg Precision: %.4f | MCC: %.4f | Kappa: %.4f | KS: %.4f",
                auc, ap, mcc, kappa, ks)

    # ─── Threshold Optimization ───
    best_f1_thresh, best_f1 = _optimize_threshold_f1(y_test, y_prob)
    best_cost_thresh, min_cost = _optimize_threshold_cost(y_test, y_prob)

    logger.info("Optimal F1 threshold: %.2f (F1 = %.4f)", best_f1_thresh, best_f1)
    logger.info(
        f"Optimal business-cost threshold: {best_cost_thresh:.2f} (min expected loss = ${min_cost:,.2f})"
    )
    logger.info(
        f"  Cost assumptions: FN = ${COST_FALSE_NEGATIVE:,.0f}, FP = ${COST_FALSE_POSITIVE:,.0f}"
    )

    # Save optimal threshold alongside model
    thresholds_info = {
        "best_f1_threshold": best_f1_thresh,
        "best_f1_score": best_f1,
        "best_cost_threshold": best_cost_thresh,
        "min_expected_cost": min_cost,
        "cost_fn": COST_FALSE_NEGATIVE,
        "cost_fp": COST_FALSE_POSITIVE,
    }
    joblib.dump(thresholds_info, m_dir / "optimal_thresholds.pkl")

    # ─── Save All Metrics as JSON ───
    all_metrics: Dict[str, Any] = {
        "model": best_model_name,
        "ROC_AUC": auc,
        "Average_Precision": round(ap, 4),
        "MCC": round(mcc, 4),
        "Cohens_Kappa": round(kappa, 4),
        "KS_Statistic": ks,
        "Confusion_Matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "Threshold_Optimization": thresholds_info,
    }
    with open(r_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved evaluation_metrics.json")

    # ─── Plots ───
    _plot_confusion_matrix(y_test, y_pred, best_model_name, i_dir / "06_confusion_matrix.png")
    _plot_roc_curve(y_test, y_prob, best_model_name, i_dir / "07_roc_curve.png")
    _plot_pr_curve(y_test, y_prob, best_model_name, i_dir / "07b_pr_curve.png")
    _plot_calibration_curve(y_test, y_prob, best_model_name, i_dir / "07c_calibration_curve.png")
    _plot_ks_chart(y_test, y_prob, best_model_name, i_dir / "07d_ks_chart.png")

    # ─── Feature Importance ───
    # Extract underlying model for feature importance
    underlying = best_model
    if hasattr(underlying, "estimator"):
        underlying = underlying.estimator
    if hasattr(underlying, "named_steps"):
        underlying = underlying.named_steps.get("model", underlying)

    if hasattr(underlying, "feature_importances_"):
        importances = underlying.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        fi_df.to_csv(d_dir / "feature_importance.csv", index=False)
        fi_df.to_csv(r_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi_df.head(15), x="Importance", y="Feature", palette="crest")
        plt.title(f"Top 15 Feature Importances ({best_model_name})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(i_dir / "08_feature_importance.png", dpi=300)
        plt.close()
        logger.info("Saved 08_feature_importance.png & feature_importance.csv")

    elif hasattr(underlying, "coef_"):
        coefs = np.abs(underlying.coef_[0])
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coefs,
        }).sort_values(by="Importance", ascending=False)

        fi_df.to_csv(d_dir / "feature_importance.csv", index=False)
        fi_df.to_csv(r_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi_df.head(15), x="Importance", y="Feature", hue="Feature", legend=False, palette="crest")
        plt.title("Top 15 Features (|Coefficient|) — Logistic Regression", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(i_dir / "08_feature_importance.png", dpi=300)
        plt.close()
        logger.info("Saved 08_feature_importance.png")

    # ─── SHAP Analysis ───
    try:
        _run_shap_analysis(
            best_model, best_model_name, feature_names,
            pred_df, i_dir, m_dir,
        )
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)

    # ─── Credit Risk Scorecard ───
    logger.info("Generating Credit Risk Scorecard...")
    pred_df["RiskCategory"] = pd.cut(
        pred_df["Predicted_Probability"], bins=RISK_BINS, labels=RISK_LABELS,
    )

    risk_summary = pred_df.groupby("RiskCategory", observed=False).agg(
        Total_Applicants=("Actual_Default", "count"),
        Actual_Defaults=("Actual_Default", "sum"),
        Actual_Default_Rate=("Actual_Default", "mean"),
    ).reset_index()
    risk_summary["Actual_Default_Rate"] = (risk_summary["Actual_Default_Rate"] * 100).round(2)

    logger.info("\n--- CREDIT RISK SCORECARD ---\n%s", risk_summary.to_string(index=False))
    risk_summary.to_csv(d_dir / "risk_scorecard_summary.csv", index=False)

    _plot_risk_scorecard(risk_summary, i_dir / "10_risk_scorecard.png")

    logger.info("Model Evaluation & Explainability completed successfully!")


if __name__ == "__main__":
    evaluate_and_explain("models", "reports", "images", "dashboard")
