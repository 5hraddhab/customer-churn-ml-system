"""
Customer Churn Prediction Module
Performs real-time and batch customer churn inference using the saved model pipeline.
Computes churn probabilities, configurable risk tiers, and key influencing factors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor


def classify_risk_tier(
    churn_prob: float,
    low_threshold: float = 0.35,
    high_threshold: float = 0.70,
) -> str:
    """
    Categorizes churn risk into Low, Medium, or High risk tiers
    based on configurable probability thresholds.
    """
    if churn_prob < low_threshold:
        return "Low Risk"
    elif churn_prob < high_threshold:
        return "Medium Risk"
    else:
        return "High Risk"


def explain_prediction_factors(
    customer_dict: Dict[str, Any],
    preprocessor: DataPreprocessor,
    model: Any,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Identifies the most influential factors for a customer's churn prediction
    using model feature importances and normalized customer feature magnitudes.
    """
    feature_names = preprocessor.get_feature_names()
    if not feature_names:
        return []

    # Get model importance weights
    weights = np.zeros(len(feature_names))
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        weights = np.abs(model.coef_[0])

    # Transform single customer record
    df_single = pd.DataFrame([customer_dict])
    try:
        transformed_row = preprocessor.transform(df_single)[0]
    except Exception:
        return []

    # Calculate contribution score: magnitude in transformed space * importance weight
    scores = np.abs(transformed_row) * weights

    # Pair features with scores
    ranked_indices = np.argsort(scores)[::-1]
    factors = []

    for idx in ranked_indices[:top_n]:
        feat = feature_names[idx]
        val = transformed_row[idx]
        # Match back to original feature name if one-hot encoded
        orig_feature = feat.split("_")[0] if "_" in feat and feat not in customer_dict else feat
        raw_val = customer_dict.get(orig_feature, customer_dict.get(feat, "N/A"))

        factors.append({
            "feature": feat,
            "raw_feature": orig_feature,
            "customer_value": raw_val,
            "impact_score": float(scores[idx]),
            "feature_weight": float(weights[idx]),
        })

    return factors


def predict_single_customer(
    pipeline_bundle: Dict[str, Any],
    customer_dict: Dict[str, Any],
    low_threshold: float = 0.35,
    high_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Generates churn prediction and risk profile for a single customer record.
    Never hardcodes outcomes; evaluates directly via the trained model pipeline.
    """
    model = pipeline_bundle["model"]
    preprocessor: DataPreprocessor = pipeline_bundle["preprocessor"]

    df_single = pd.DataFrame([customer_dict])
    X_trans = preprocessor.transform(df_single)

    # Churn probability
    churn_prob = 0.5
    has_prob = hasattr(model, "predict_proba")
    if has_prob:
        probs = model.predict_proba(X_trans)[0]
        # Class 1 is churn
        if len(probs) >= 2:
            churn_prob = float(probs[1])
        else:
            churn_prob = float(probs[0])
        pred_label = 1 if churn_prob >= 0.5 else 0
    else:
        raw_pred = model.predict(X_trans)[0]
        pred_label = int(raw_pred)
        churn_prob = 1.0 if pred_label == 1 else 0.0

    risk_tier = classify_risk_tier(churn_prob, low_threshold, high_threshold)

    key_factors = explain_prediction_factors(
        customer_dict=customer_dict,
        preprocessor=preprocessor,
        model=model,
        top_n=5,
    )

    return {
        "predicted_label": pred_label,
        "predicted_status": "Churned" if pred_label == 1 else "Retained",
        "churn_probability": churn_prob,
        "retention_probability": 1.0 - churn_prob,
        "risk_tier": risk_tier,
        "model_used": pipeline_bundle.get("model_name", "Trained Model"),
        "key_factors": key_factors,
    }


def predict_batch_customers(
    pipeline_bundle: Dict[str, Any],
    batch_df: pd.DataFrame,
    low_threshold: float = 0.35,
    high_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Executes batch inference across a customer DataFrame, returning
    the input DataFrame enriched with Churn_Prediction, Churn_Probability, and Risk_Tier.
    """
    model = pipeline_bundle["model"]
    preprocessor: DataPreprocessor = pipeline_bundle["preprocessor"]

    result_df = batch_df.copy()
    X_trans = preprocessor.transform(result_df)

    # Predictions
    has_prob = hasattr(model, "predict_proba")
    if has_prob:
        probs = model.predict_proba(X_trans)
        if probs.shape[1] >= 2:
            churn_probs = probs[:, 1]
        else:
            churn_probs = probs[:, 0]
        predictions = (churn_probs >= 0.5).astype(int)
    else:
        predictions = model.predict(X_trans)
        churn_probs = predictions.astype(float)

    result_df["Churn_Prediction"] = np.where(predictions == 1, "Churn", "Retained")
    result_df["Churn_Probability"] = np.round(churn_probs, 4)
    result_df["Risk_Tier"] = [
        classify_risk_tier(p, low_threshold, high_threshold) for p in churn_probs
    ]

    return result_df
