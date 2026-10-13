"""
Adaptive Analytics Dashboard Module
Generates Plotly interactive visualizations dynamically adapted to any dataset schema.
Strictly verifies that columns exist before attempting visualization.
Distinguishes historical churn from predicted churn.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def compute_overview_metrics(
    df: pd.DataFrame,
    target_col: str,
    churn_val: Any = 1,
    retained_val: Any = 0,
) -> Dict[str, Any]:
    """
    Computes key baseline metrics from historical dataset.
    """
    if target_col not in df.columns:
        return {
            "total_customers": len(df),
            "churned_count": 0,
            "retained_count": 0,
            "churn_rate": 0.0,
            "retention_rate": 0.0,
        }

    total = len(df)
    target_series = df[target_col]

    churn_mask = target_series.astype(str).str.strip().str.lower() == str(churn_val).strip().lower()
    churn_count = int(churn_mask.sum())
    retained_count = total - churn_count
    churn_rate = (churn_count / total * 100) if total > 0 else 0.0
    retention_rate = 100.0 - churn_rate

    return {
        "total_customers": total,
        "churned_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": round(churn_rate, 2),
        "retention_rate": round(retention_rate, 2),
    }


def plot_churn_donut(
    df: pd.DataFrame,
    target_col: str,
    churn_val: Any = 1,
    retained_val: Any = 0,
) -> go.Figure:
    """
    Interactive donut chart of historical churn distribution.
    """
    metrics = compute_overview_metrics(df, target_col, churn_val, retained_val)
    labels = ["Retained Customers", "Churned Customers"]
    values = [metrics["retained_count"], metrics["churned_count"]]
    colors = ["#22c55e", "#ef4444"]  # Green for retained, Red for churn

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent+value",
            hoverinfo="label+value+percent",
        )
    ])
    fig.update_layout(
        title="Historical Churn Class Distribution",
        template="plotly_white",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def plot_feature_vs_churn(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    churn_val: Any = 1,
    retained_val: Any = 0,
) -> Optional[go.Figure]:
    """
    Dynamically renders feature distribution grouped by Churn vs Retained.
    Adapts to whether feature is numerical or categorical.
    """
    if feature_col not in df.columns or target_col not in df.columns:
        return None

    plot_df = df[[feature_col, target_col]].dropna().copy()
    if plot_df.empty:
        return None

    # Binary label mapping for readability
    churn_mask = plot_df[target_col].astype(str).str.strip().str.lower() == str(churn_val).strip().lower()
    plot_df["Churn_Label"] = np.where(churn_mask, "Churned", "Retained")

    color_map = {"Retained": "#22c55e", "Churned": "#ef4444"}

    # Numerical feature -> Box plot & Violin/Histogram
    if pd.api.types.is_numeric_dtype(plot_df[feature_col]):
        fig = px.box(
            plot_df,
            x="Churn_Label",
            y=feature_col,
            color="Churn_Label",
            color_discrete_map=color_map,
            points="outliers",
            title=f"Distribution of '{feature_col}' by Churn Status",
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_title="Customer Status",
            yaxis_title=feature_col,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return fig
    else:
        # Categorical feature -> Grouped percentage bar chart
        ct = pd.crosstab(plot_df[feature_col], plot_df["Churn_Label"], normalize="index") * 100
        ct = ct.reset_index()

        fig = go.Figure()
        if "Retained" in ct.columns:
            fig.add_trace(go.Bar(
                name="Retained (%)",
                x=ct[feature_col].astype(str),
                y=ct["Retained"],
                marker_color="#22c55e",
            ))
        if "Churned" in ct.columns:
            fig.add_trace(go.Bar(
                name="Churned (%)",
                x=ct[feature_col].astype(str),
                y=ct["Churned"],
                marker_color="#ef4444",
            ))

        fig.update_layout(
            barmode="group",
            title=f"Churn Proportion across '{feature_col}' Categories",
            xaxis_title=feature_col,
            yaxis_title="Percentage (%)",
            template="plotly_white",
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        return fig


def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str]) -> Optional[go.Figure]:
    """
    Renders correlation matrix strictly for existing numerical columns.
    """
    valid_cols = [c for c in numerical_cols if c in df.columns]
    if len(valid_cols) < 2:
        return None

    corr = df[valid_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numerical Features Correlation Heatmap",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def plot_model_comparison(evaluation_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Bar chart comparing models across Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    records = []
    for model_name, metrics in evaluation_results.items():
        records.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "Precision (Churn)": metrics.get("precision", 0.0),
            "Recall (Churn)": metrics.get("recall", 0.0),
            "F1-Score (Churn)": metrics.get("f1", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0) or 0.0,
        })

    comp_df = pd.DataFrame(records)
    melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Comprehensive Model Performance Comparison",
        color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"],
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 1.05],
        yaxis_title="Score",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def plot_confusion_matrix_interactive(cm: List[List[int]], model_name: str) -> go.Figure:
    """
    Interactive confusion matrix with TN, FP, FN, TP labels.
    """
    cm_arr = np.array(cm)
    labels = [["True Retained (TN)", "False Churn (FP)"],
              ["False Retained (FN)", "True Churn (TP)"]]

    annotations_text = [
        [f"{labels[i][j]}<br><b>{cm_arr[i][j]}</b>" for j in range(2)]
        for i in range(2)
    ]

    fig = ff_cm = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=["Predicted Retained", "Predicted Churned"],
        y=["Actual Retained", "Actual Churned"],
        hoverongaps=False,
        colorscale="Blues",
        text=annotations_text,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        template="plotly_white",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
    """
    Horizontal bar chart of top predictive features.
    """
    if importance_df.empty:
        return None

    top_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["importance"],
        y=top_df["feature"],
        orientation="h",
        marker=dict(color="#3b82f6"),
        text=top_df["importance"].round(3),
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Top {top_n} Predictive Feature Importances",
        template="plotly_white",
        xaxis_title="Relative Importance Weight",
        yaxis_title="Feature",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def plot_predicted_risk_distribution(batch_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Renders risk tier breakdown from batch prediction outputs.
    """
    if "Risk_Tier" not in batch_df.columns:
        return None

    counts = batch_df["Risk_Tier"].value_counts().reset_index()
    counts.columns = ["Risk Tier", "Count"]

    color_map = {
        "Low Risk": "#22c55e",
        "Medium Risk": "#f59e0b",
        "High Risk": "#ef4444",
    }

    fig = px.bar(
        counts,
        x="Risk Tier",
        y="Count",
        color="Risk Tier",
        color_discrete_map=color_map,
        text="Count",
        title="Predicted Customer Risk Tier Distribution",
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig
