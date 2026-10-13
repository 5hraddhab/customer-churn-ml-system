"""
AI-Based Customer Churn Prediction and Customer Retention System
Streamlit Multi-Page Web Application
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    load_dataset,
    get_default_dataset_path,
    detect_column_types,
    detect_id_columns,
    validate_target_column,
    inspect_dataset,
)
from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer
from src.predict import (
    predict_single_customer,
    predict_batch_customers,
    classify_risk_tier,
)
from src.analytics import (
    compute_overview_metrics,
    plot_churn_donut,
    plot_feature_vs_churn,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_confusion_matrix_interactive,
    plot_feature_importance_bar,
    plot_predicted_risk_distribution,
)
from src.recommendations import (
    INDUSTRY_PROFILES,
    generate_retention_recommendations,
)

# Page Configuration
st.set_page_config(
    page_title="AI Customer Churn & Retention System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    .status-badge-churn {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-retained {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-high {
        color: #dc2626;
        font-weight: 700;
    }
    .risk-med {
        color: #d97706;
        font-weight: 700;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "df" not in st.session_state:
        # Load default dataset automatically if available
        default_path = get_default_dataset_path()
        if default_path:
            try:
                st.session_state.df = load_dataset(default_path)
                st.session_state.dataset_source = f"Default Raw Dataset ({os.path.basename(default_path)})"
            except Exception:
                st.session_state.df = None
                st.session_state.dataset_source = None
        else:
            st.session_state.df = None
            st.session_state.dataset_source = None

    if "target_col" not in st.session_state:
        st.session_state.target_col = "Churn" if (st.session_state.df is not None and "Churn" in st.session_state.df.columns) else None
    if "churn_val" not in st.session_state:
        st.session_state.churn_val = 1
    if "retained_val" not in st.session_state:
        st.session_state.retained_val = 0
    if "excluded_cols" not in st.session_state:
        if st.session_state.df is not None:
            st.session_state.excluded_cols = detect_id_columns(st.session_state.df)
        else:
            st.session_state.excluded_cols = []
    if "pipeline_bundle" not in st.session_state:
        # Check if saved model exists on disk
        saved_model_path = os.path.join("models", "best_model.joblib")
        if os.path.exists(saved_model_path):
            try:
                st.session_state.pipeline_bundle = ModelTrainer.load_pipeline(saved_model_path)
            except Exception:
                st.session_state.pipeline_bundle = None
        else:
            st.session_state.pipeline_bundle = None
    if "batch_predictions" not in st.session_state:
        st.session_state.batch_predictions = None
    if "selected_industry" not in st.session_state:
        st.session_state.selected_industry = "General / Configurable"


init_session_state()


# Sidebar Navigation
st.sidebar.title("🧭 System Navigation")
pages = [
    "Home",
    "Upload Dataset",
    "Data Analysis",
    "Train Model",
    "Predict Churn",
    "Customer Insights",
    "Model Performance",
    "Retention Recommendations",
]
choice = st.sidebar.radio("Go to Page:", pages)

st.sidebar.markdown("---")
# Active Dataset Status in Sidebar
if st.session_state.df is not None:
    st.sidebar.success(f"📁 **Active Data:** {st.session_state.get('dataset_source', 'Custom')}")
    st.sidebar.caption(f"Rows: {len(st.session_state.df):,} | Columns: {st.session_state.df.shape[1]}")
    if st.session_state.target_col:
        st.sidebar.caption(f"Target: `{st.session_state.target_col}` (Churn={st.session_state.churn_val})")
else:
    st.sidebar.warning("⚠️ No dataset currently loaded.")

# Active Model Status in Sidebar
if st.session_state.pipeline_bundle is not None:
    model_name = st.session_state.pipeline_bundle.get("model_name", "Trained Model")
    st.sidebar.info(f"🤖 **Active Model:** {model_name}")
else:
    st.sidebar.caption("🤖 Model: Not yet trained/loaded")

st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Customer Churn Prediction & Retention System | AIML Project")


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if choice == "Home":
    st.markdown('<div class="main-header">AI-Based Customer Churn Prediction & Retention System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Generalized, Multi-Industry Machine Learning Framework for Proactive Customer Attrition Mitigation</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records Loaded", f"{len(st.session_state.df):,}" if st.session_state.df is not None else "0")
    with col2:
        st.metric("Features Count", f"{st.session_state.df.shape[1]}" if st.session_state.df is not None else "0")
    with col3:
        st.metric("Target Column", str(st.session_state.target_col) if st.session_state.target_col else "Not Set")
    with col4:
        best_name = st.session_state.pipeline_bundle.get("model_name", "None") if st.session_state.pipeline_bundle else "None"
        st.metric("Active ML Model", best_name)

    st.markdown("### 🎯 System Architecture & Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 1. Generalized Architecture
        - **Cross-Industry Reusability:** Accommodates banking, retail, telecoms, SaaS, and subscription business models.
        - **Flexible Schema Detection:** Dynamically recognizes numerical, categorical, and identifier columns without hardcoded dependencies.
        - **Configurable Target:** Supports arbitrary churn/retention labels (e.g. 1/0, Yes/No, True/False).
        """)
    with c2:
        c2.markdown("""
        #### 2. Robust ML Pipeline
        - **Zero Data Leakage:** Preprocessing transforms are strictly fitted only on the training split.
        - **Tri-Model Benchmarking:** Compares Logistic Regression, Decision Tree, and Random Forest.
        - **Class-Imbalance Aware:** Optimized for recall and F1-score to detect subtle churn signals.
        """)
    with c3:
        c3.markdown("""
        #### 3. Actionable Retention Engine
        - **Real-Time Inference:** Generates churn probabilities and calibrated risk tiers (Low, Medium, High).
        - **Factor Attribution:** Highlights the top customer behavioral drivers associated with churn.
        - **Industry-Specific Playbooks:** Recommends targeted, explainable retention interventions with documented rationales.
        """)

    st.markdown("---")
    st.markdown("### 🚀 Guided Workflow")
    st.info("""
    1. **Upload Dataset:** Connect your custom CSV/Excel dataset or use the pre-loaded default dataset.
    2. **Data Analysis:** Audit data quality, missing values, duplicates, and feature distributions.
    3. **Train Model:** Configure scaling, train Logistic Regression, Decision Tree & Random Forest, and compare test results.
    4. **Predict Churn:** Enter customer characteristics for instant risk scores or upload a batch file.
    5. **Customer Insights:** Filter high-risk cohorts and analyze attrition distributions.
    6. **Retention Recommendations:** Review tailored, industry-aligned retention strategies and action steps.
    """)


# ==============================================================================
# PAGE 2: UPLOAD DATASET
# ==============================================================================
elif choice == "Upload Dataset":
    st.markdown('<div class="main-header">Generalized Dataset Upload & Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload arbitrary CSV or Excel files, map target churn columns, and configure feature exclusions.</div>', unsafe_allow_html=True)

    tab_upload, tab_default = st.tabs(["📤 Upload New Dataset", "📁 Default Dataset"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload Customer Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                new_df = load_dataset(uploaded_file)
                st.session_state.df = new_df
                st.session_state.dataset_source = f"Uploaded File ({uploaded_file.name})"
                st.session_state.excluded_cols = detect_id_columns(new_df)
                st.success(f"Successfully loaded '{uploaded_file.name}' with {len(new_df):,} rows and {new_df.shape[1]} columns!")
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")

    with tab_default:
        default_path = get_default_dataset_path()
        if default_path:
            st.write(f"Default dataset available at: `{default_path}`")
            if st.button("🔄 Reset / Reload Default Dataset"):
                st.session_state.df = load_dataset(default_path)
                st.session_state.dataset_source = f"Default Raw Dataset ({os.path.basename(default_path)})"
                st.session_state.target_col = "Churn"
                st.session_state.churn_val = 1
                st.session_state.retained_val = 0
                st.session_state.excluded_cols = detect_id_columns(st.session_state.df)
                st.success("Default dataset loaded successfully!")
        else:
            st.warning("Default dataset not found in `Data/` folder.")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.subheader("⚙️ Target Column & Identifier Configuration")

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            all_cols = list(df.columns)
            target_idx = all_cols.index(st.session_state.target_col) if st.session_state.target_col in all_cols else len(all_cols) - 1
            selected_target = st.selectbox("Select Target Churn Column:", all_cols, index=target_idx)
            st.session_state.target_col = selected_target

        # Validate target column
        target_info = validate_target_column(df, selected_target)

        with col_t2:
            unique_vals = target_info.get("unique_values", [])
            churn_def = target_info.get("suggested_churn_val", unique_vals[0] if unique_vals else 1)
            churn_idx = unique_vals.index(churn_def) if churn_def in unique_vals else 0
            selected_churn_val = st.selectbox("Value representing CHURN (1):", unique_vals, index=churn_idx if unique_vals else 0)
            st.session_state.churn_val = selected_churn_val

        with col_t3:
            retained_def = target_info.get("suggested_retained_val", unique_vals[-1] if len(unique_vals) > 1 else 0)
            ret_idx = unique_vals.index(retained_def) if retained_def in unique_vals else (1 if len(unique_vals) > 1 else 0)
            selected_ret_val = st.selectbox("Value representing RETAINED (0):", unique_vals, index=ret_idx if unique_vals else 0)
            st.session_state.retained_val = selected_ret_val

        # Display target validation feedback
        if not target_info["valid"]:
            for err in target_info["errors"]:
                st.error(f"❌ Target Error: {err}")
        else:
            st.success(f"✅ Valid binary classification target: Churn = '{selected_churn_val}', Retained = '{selected_ret_val}'")
            if target_info["warnings"]:
                for w in target_info["warnings"]:
                    st.warning(f"⚠️ {w}")

        st.markdown("#### 🚫 Feature Exclusion & Identifiers")
        st.caption("Identifiers (e.g. Customer_ID) and irrelevant features should be excluded to prevent data contamination.")
        candidate_ids = detect_id_columns(df)
        default_excluded = [c for c in candidate_ids if c != selected_target]

        other_cols = [c for c in df.columns if c != selected_target]
        selected_excluded = st.multiselect(
            "Columns to exclude from model training:",
            options=other_cols,
            default=[c for c in st.session_state.excluded_cols if c in other_cols] or default_excluded,
        )
        st.session_state.excluded_cols = selected_excluded

        st.markdown("---")
        st.subheader("📋 Dataset Preview (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)


# ==============================================================================
# PAGE 3: DATA ANALYSIS
# ==============================================================================
elif choice == "Data Analysis":
    st.markdown('<div class="main-header">Automatic Dataset Inspection & Quality Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive statistical inspection, missing value profiling, and adaptive feature distributions.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset or load the default dataset in 'Upload Dataset'.")
    else:
        df = st.session_state.df
        target_col = st.session_state.target_col
        churn_val = st.session_state.churn_val
        retained_val = st.session_state.retained_val

        # Perform inspection
        inspection = inspect_dataset(df, target_col=target_col, churn_value=churn_val, retained_value=retained_val)

        # Overview metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Records", f"{inspection['n_rows']:,}")
        m2.metric("Total Columns", f"{inspection['n_cols']}")
        m3.metric("Duplicate Rows", f"{inspection['duplicate_rows']}")
        m4.metric("Columns w/ Missing", f"{inspection['columns_with_missing_count']}")
        m5.metric("Detected ID Columns", f"{len(inspection['candidate_ids'])}")

        # Data Quality Alert Box
        st.markdown("### 🛡️ Data Quality Audit")
        if inspection["quality_issues"]:
            for issue in inspection["quality_issues"]:
                st.warning(f"⚠️ {issue['message']}")
        else:
            st.success("✅ Clean dataset: No missing values, duplicate rows, or extreme class skew detected.")

        st.markdown("---")
        # Churn Class Distribution & Summary
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            if target_col in df.columns:
                fig_donut = plot_churn_donut(df, target_col, churn_val, retained_val)
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Target column not specified.")

        with col_c2:
            st.markdown("#### 📊 Target Breakdown Statistics")
            if target_col in df.columns:
                overview = compute_overview_metrics(df, target_col, churn_val, retained_val)
                st.write(f"- **Total Customers Analyzed:** {overview['total_customers']:,}")
                st.write(f"- **Historical Churned Count:** {overview['churned_count']:,} ({overview['churn_rate']}%)")
                st.write(f"- **Historical Retained Count:** {overview['retained_count']:,} ({overview['retention_rate']}%)")
                st.write(f"- **Baseline Imbalance Ratio:** 1 Churned for every {round(overview['retained_count']/max(overview['churned_count'], 1), 2)} Retained")

            st.markdown("#### 🔍 Detected Column Types")
            col_types = inspection["column_types"]
            st.write(f"- **Numerical Features ({len(col_types['numerical'])}):** {', '.join(col_types['numerical'][:6])}{'...' if len(col_types['numerical']) > 6 else ''}")
            st.write(f"- **Categorical Features ({len(col_types['categorical'])}):** {', '.join(col_types['categorical'])}")
            if inspection["candidate_ids"]:
                st.write(f"- **Candidate Identifier(s):** {', '.join(inspection['candidate_ids'])}")

        st.markdown("---")
        # Dynamic Feature Explorer
        st.markdown("### 📈 Adaptive Feature vs. Churn Visualizer")
        st.caption("Select any feature from your dataset to inspect how its values distribute between Churned and Retained customers.")

        available_features = [c for c in df.columns if c != target_col and c not in st.session_state.excluded_cols]
        if available_features:
            selected_feat = st.selectbox("Select Feature to Analyze:", available_features, index=0)
            fig_feat = plot_feature_vs_churn(df, selected_feat, target_col, churn_val, retained_val)
            if fig_feat:
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.warning("Could not generate visualization for the selected feature.")
        else:
            st.info("No feature columns available to visualize.")

        # Correlation Matrix
        num_cols = inspection["column_types"]["numerical"]
        valid_num_cols = [c for c in num_cols if c not in st.session_state.excluded_cols and c in df.columns]
        if len(valid_num_cols) >= 2:
            st.markdown("---")
            st.markdown("### 🧬 Numerical Correlation Heatmap")
            fig_corr = plot_correlation_heatmap(df, valid_num_cols)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# PAGE 4: TRAIN MODEL
# ==============================================================================
elif choice == "Train Model":
    st.markdown('<div class="main-header">Configurable Machine Learning Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train and benchmark Logistic Regression, Decision Tree, and Random Forest without data leakage.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload or load a dataset first.")
    elif not st.session_state.target_col:
        st.error("Please configure a valid target column in 'Upload Dataset'.")
    else:
        df = st.session_state.df

        st.subheader("⚙️ Preprocessing & Model Hyperparameters")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            scaler_choice = st.selectbox("Numerical Scaler:", ["standard", "minmax", "robust", "none"], index=0)
            test_split = st.slider("Test Set Split Ratio:", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        with c_p2:
            cat_encoder = st.selectbox("Categorical Encoder:", ["onehot", "ordinal"], index=0)
            handle_imbalance = st.checkbox("Handle Class Imbalance (Balanced Weights)", value=True)
        with c_p3:
            selection_metric = st.selectbox(
                "Primary Model Selection Metric:",
                ["f1", "recall", "accuracy", "roc_auc"],
                index=0,
                help="F1-score balances precision and recall. Recall prioritizes catching as many churners as possible.",
            )

        st.info("💡 **Leakage Prevention:** All scalers, encoders, and imputers are strictly fitted on the training split only (`X_train`) and transformed onto `X_test`.")

        if st.button("🚀 Train & Benchmark Models", type="primary"):
            with st.spinner("Training Logistic Regression, Decision Tree, and Random Forest..."):
                try:
                    preprocessor = DataPreprocessor(
                        target_col=st.session_state.target_col,
                        churn_value=st.session_state.churn_val,
                        retained_value=st.session_state.retained_val,
                        excluded_columns=st.session_state.excluded_cols,
                        scaler=scaler_choice,
                        categorical_encoder=cat_encoder,
                    )

                    trainer = ModelTrainer(
                        test_size=test_split,
                        selection_metric=selection_metric,
                        handle_imbalance=handle_imbalance,
                    )

                    train_out = trainer.train_and_evaluate(df, preprocessor)

                    # Save pipeline
                    save_path = os.path.join("models", "best_model.joblib")
                    trainer.save_pipeline(save_path, preprocessor, metadata={
                        "scaler": scaler_choice,
                        "selection_metric": selection_metric,
                        "excluded_cols": st.session_state.excluded_cols,
                    })

                    st.session_state.pipeline_bundle = ModelTrainer.load_pipeline(save_path)
                    st.success(f"🎉 Training complete! Best performing model: **{train_out['best_model_name']}** (Optimized for {selection_metric.upper()}).")

                except Exception as e:
                    st.error(f"Training failed: {e}")

        # Display results if pipeline is trained
        if st.session_state.pipeline_bundle is not None:
            bundle = st.session_state.pipeline_bundle
            eval_results = bundle.get("evaluation_results", {})
            best_model_name = bundle.get("model_name", "Trained Model")

            st.markdown("---")
            st.markdown(f"### 🏆 Model Comparison Matrix (Best: **{best_model_name}**)")

            summary_rows = []
            for name, m in eval_results.items():
                summary_rows.append({
                    "Model": name,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Precision (Churn)": f"{m['precision']:.4f}",
                    "Recall (Churn)": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1']:.4f}",
                    "ROC-AUC": f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A",
                    "Status": "⭐ Selected Best" if name == best_model_name else "Candidate",
                })

            st.table(pd.DataFrame(summary_rows))

            # Model comparison bar chart
            fig_comp = plot_model_comparison(eval_results)
            st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# PAGE 5: PREDICT CHURN
# ==============================================================================
elif choice == "Predict Churn":
    st.markdown('<div class="main-header">Real-Time & Batch Customer Churn Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate predictions using the fitted ML pipeline with calibrated risk tiers and top behavioral factors.</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_bundle is None:
        st.warning("⚠️ No trained model pipeline found. Please train a model in the 'Train Model' page first.")
    else:
        bundle = st.session_state.pipeline_bundle
        preprocessor: DataPreprocessor = bundle["preprocessor"]
        active_model = bundle.get("model_name", "Trained Model")

        tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📂 Batch Customer Scoring"])

        # TAB 1: SINGLE CUSTOMER INFERENCE
        with tab_single:
            st.subheader("Customer Characteristics Input Form")
            st.caption(f"Inference using active pipeline: **{active_model}**")

            # Extract features from preprocessor
            num_cols = preprocessor.fitted_numerical_cols
            cat_cols = preprocessor.fitted_categorical_cols

            customer_inputs = {}

            # Create 2 columns for numerical and categorical inputs
            col_in1, col_in2 = st.columns(2)

            with col_in1:
                st.markdown("##### 🔢 Numerical Features")
                for col in num_cols:
                    # Determine reasonable defaults from dataset if available
                    col_series = st.session_state.df[col] if (st.session_state.df is not None and col in st.session_state.df.columns) else None
                    if col_series is not None:
                        min_v = float(col_series.min())
                        max_v = float(col_series.max())
                        med_v = float(col_series.median())
                        # If integer, use step 1
                        if pd.api.types.is_integer_dtype(col_series):
                            customer_inputs[col] = st.number_input(
                                f"{col}:",
                                min_value=int(min_v),
                                max_value=int(max_v),
                                value=int(med_v),
                                step=1,
                            )
                        else:
                            customer_inputs[col] = st.number_input(
                                f"{col}:",
                                min_value=float(min_v),
                                max_value=float(max_v),
                                value=float(med_v),
                                step=0.1,
                            )
                    else:
                        customer_inputs[col] = st.number_input(f"{col}:", value=0.0)

            with col_in2:
                st.markdown("##### 🏷️ Categorical Features")
                for col in cat_cols:
                    col_series = st.session_state.df[col] if (st.session_state.df is not None and col in st.session_state.df.columns) else None
                    if col_series is not None:
                        unique_options = sorted([str(v) for v in col_series.dropna().unique().tolist()])
                        customer_inputs[col] = st.selectbox(f"{col}:", options=unique_options, index=0)
                    else:
                        customer_inputs[col] = st.text_input(f"{col}:", value="Standard")

            # Risk Thresholds
            st.markdown("---")
            with st.expander("🛠️ Configure Risk Tier Thresholds"):
                r1, r2 = st.columns(2)
                low_th = r1.slider("Low-Risk Threshold (Below = Low Risk):", 0.10, 0.50, 0.35, 0.05)
                high_th = r2.slider("High-Risk Threshold (Above = High Risk):", 0.50, 0.90, 0.70, 0.05)

            if st.button("🔮 Calculate Churn Prediction", type="primary"):
                try:
                    pred_res = predict_single_customer(
                        pipeline_bundle=bundle,
                        customer_dict=customer_inputs,
                        low_threshold=low_th,
                        high_threshold=high_th,
                    )

                    st.markdown("---")
                    st.subheader("🎯 Prediction & Risk Diagnostic")

                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        if pred_res["predicted_label"] == 1:
                            st.markdown("### Status: <span class='status-badge-churn'>CHURN RISK</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("### Status: <span class='status-badge-retained'>RETAINED</span>", unsafe_allow_html=True)

                    with p2:
                        st.metric("Churn Probability", f"{pred_res['churn_probability'] * 100:.1f}%")

                    with p3:
                        st.metric("Retention Probability", f"{pred_res['retention_probability'] * 100:.1f}%")

                    with p4:
                        tier = pred_res["risk_tier"]
                        css_class = "risk-high" if "High" in tier else ("risk-med" if "Med" in tier else "risk-low")
                        st.markdown(f"### Tier: <span class='{css_class}'>{tier}</span>", unsafe_allow_html=True)

                    # Top Influencing Factors
                    st.markdown("#### 🔍 Top Factors Influencing Prediction")
                    if pred_res["key_factors"]:
                        factor_rows = []
                        for f in pred_res["key_factors"]:
                            factor_rows.append({
                                "Feature": f["raw_feature"],
                                "Customer Value": f["customer_value"],
                                "Model Weight": round(f["feature_weight"], 4),
                                "Attribution Score": round(f["impact_score"], 4),
                            })
                        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True)

                    # Retention Action Preview
                    st.markdown("#### 💡 Suggested Retention Interventions")
                    recs = generate_retention_recommendations(
                        customer_dict=customer_inputs,
                        prediction_result=pred_res,
                        industry=st.session_state.selected_industry,
                    )
                    for r in recs:
                        st.markdown(f"**[{r['priority']} Priority] {r['title']}**")
                        st.write(f"*{r['rationale']}*")
                        st.info(f"👉 **Action:** {r['suggested_action']}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # TAB 2: BATCH INFERENCE
        with tab_batch:
            st.subheader("Batch Scoring on New Customer Cohorts")
            st.caption("Upload a CSV file or score the currently loaded dataset.")

            score_choice = st.radio("Choose Batch Data Source:", ["Score Current Dataset", "Upload Separate CSV"])

            batch_to_score = None
            if score_choice == "Score Current Dataset":
                if st.session_state.df is not None:
                    batch_to_score = st.session_state.df
                else:
                    st.warning("No current dataset loaded.")
            else:
                batch_file = st.file_uploader("Upload Batch CSV for Scoring", type=["csv"], key="batch_uploader")
                if batch_file:
                    batch_to_score = load_dataset(batch_file)

            if batch_to_score is not None:
                st.write(f"Ready to score **{len(batch_to_score):,}** records.")
                if st.button("⚡ Run Batch Scoring"):
                    with st.spinner("Scoring customer records..."):
                        scored_df = predict_batch_customers(bundle, batch_to_score)
                        st.session_state.batch_predictions = scored_df
                        st.success(f"Batch scoring completed for {len(scored_df):,} customers!")

            if st.session_state.batch_predictions is not None:
                b_df = st.session_state.batch_predictions
                st.markdown("---")
                st.subheader("📊 Batch Scoring Results")

                b1, b2, b3 = st.columns(3)
                total_scored = len(b_df)
                high_risk_cnt = int((b_df["Risk_Tier"] == "High Risk").sum())
                churn_pred_cnt = int((b_df["Churn_Prediction"] == "Churn").sum())

                b1.metric("Total Scored", f"{total_scored:,}")
                b2.metric("Predicted Churners", f"{churn_pred_cnt:,} ({churn_pred_cnt/total_scored*100:.1f}%)")
                b3.metric("High-Risk Tier", f"{high_risk_cnt:,} ({high_risk_cnt/total_scored*100:.1f}%)")

                st.plotly_chart(plot_predicted_risk_distribution(b_df), use_container_width=True)

                st.dataframe(b_df.head(20), use_container_width=True)

                # Export Download
                csv_data = b_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored Batch Predictions (CSV)",
                    data=csv_data,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                )


# ==============================================================================
# PAGE 6: CUSTOMER INSIGHTS
# ==============================================================================
elif choice == "Customer Insights":
    st.markdown('<div class="main-header">Strategic Customer Insights & High-Risk Cohorts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Segment customer cohorts, analyze attrition drivers, and isolate priority accounts.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload or load a dataset first.")
    else:
        df = st.session_state.df
        target_col = st.session_state.target_col
        churn_val = st.session_state.churn_val

        st.subheader("🎯 High-Risk Customer Identification")
        st.caption("Distinguishing historical churn from model-predicted churn risks.")

        # Check if batch scored dataset is available
        if st.session_state.batch_predictions is not None:
            scored_df = st.session_state.batch_predictions
            st.markdown("#### 🚨 Predicted High-Risk Customer List")
            high_risk = scored_df[scored_df["Risk_Tier"] == "High Risk"].sort_values(by="Churn_Probability", ascending=False)
            st.write(f"Identified **{len(high_risk):,}** high-risk customers requiring urgent retention outreach.")
            st.dataframe(high_risk.head(25), use_container_width=True)
        else:
            st.info("Tip: Run batch scoring in 'Predict Churn' to generate individual risk probabilities for every customer in your dataset.")

        st.markdown("---")
        st.subheader("👥 Cohort Segmentation by Key Business Dimensions")

        cat_cols = detect_column_types(df, target_col=target_col)["categorical"]
        if cat_cols:
            col_seg1, col_seg2 = st.columns(2)
            with col_seg1:
                selected_cat = st.selectbox("Segment by Categorical Dimension:", cat_cols, index=0)
            with col_seg2:
                metric_choice = st.selectbox("Calculation:", ["Churn Rate (%)", "Total Customer Count"])

            fig_seg = plot_feature_vs_churn(df, selected_cat, target_col, churn_val, st.session_state.retained_val)
            if fig_seg:
                st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.write("No categorical dimensions found for segmentation.")


# ==============================================================================
# PAGE 7: MODEL PERFORMANCE
# ==============================================================================
elif choice == "Model Performance":
    st.markdown('<div class="main-header">Rigorous Model Performance & Evaluation Deep-Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect confusion matrices, classification reports, and predictive feature importance rankings.</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_bundle is None:
        st.warning("No trained model pipeline available. Please train models in 'Train Model' first.")
    else:
        bundle = st.session_state.pipeline_bundle
        eval_results = bundle.get("evaluation_results", {})
        all_models = bundle.get("all_models", {})
        feature_importances = bundle.get("feature_importances", {})
        best_name = bundle.get("model_name", "Trained Model")

        model_tabs = st.tabs(list(eval_results.keys()))

        for tab, (m_name, metrics) in zip(model_tabs, eval_results.items()):
            with tab:
                st.markdown(f"### {m_name} Evaluation Summary {'⭐ (Selected Best)' if m_name == best_name else ''}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                c2.metric("Precision (Churn)", f"{metrics['precision']:.4f}")
                c3.metric("Recall (Churn)", f"{metrics['recall']:.4f}")
                c4.metric("F1-Score", f"{metrics['f1']:.4f}")
                c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A")

                st.markdown("---")
                col_cm, col_cr = st.columns([1, 1])

                with col_cm:
                    st.markdown("#### 🎯 Interactive Confusion Matrix")
                    cm = metrics["confusion_matrix"]
                    fig_cm = plot_confusion_matrix_interactive(cm, m_name)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_cr:
                    st.markdown("#### 📋 Detailed Classification Report")
                    report_dict = metrics.get("classification_report", {})
                    if report_dict:
                        rep_df = pd.DataFrame(report_dict).T.round(4)
                        st.dataframe(rep_df, use_container_width=True)

                # Feature importances for this model
                if m_name in feature_importances:
                    st.markdown("---")
                    st.markdown(f"#### 🏆 Top Feature Importances ({m_name})")
                    imp_df = feature_importances[m_name]
                    fig_imp = plot_feature_importance_bar(imp_df, top_n=10)
                    if fig_imp:
                        st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# PAGE 8: RETENTION RECOMMENDATIONS
# ==============================================================================
elif choice == "Retention Recommendations":
    st.markdown('<div class="main-header">Configurable Retention Strategy Playbook</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Contextual, industry-specific retention recommendations mapped to behavioral indicators.</div>', unsafe_allow_html=True)

    st.markdown("### 🏢 Select Industry Business Profile")
    industries = list(INDUSTRY_PROFILES.keys())
    curr_idx = industries.index(st.session_state.selected_industry) if st.session_state.selected_industry in industries else 0
    selected_ind = st.selectbox("Target Business Sector:", industries, index=curr_idx)
    st.session_state.selected_industry = selected_ind

    profile = INDUSTRY_PROFILES[selected_ind]
    st.info(f"**Sector Profile:** {profile['description']}")

    st.markdown("---")
    st.markdown("### 📘 Active Strategic Retention Matrix")
    for strat in profile["strategies"]:
        with st.expander(f"📌 {strat['title']} [{strat['priority']} Priority]"):
            st.markdown(f"**Rationale & Behavioral Trigger:**\n{strat['rationale']}")
            st.markdown(f"**Recommended Action Plan:**\n{strat['action']}")

    st.markdown("---")
    st.markdown("### 🧪 Scenario Simulator: Test Retention Recommendations")
    st.caption("Input hypothetical customer signals to preview the triggered retention recommendations.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_risk = st.selectbox("Simulated Risk Tier:", ["High Risk", "Medium Risk", "Low Risk"], index=0)
        sim_spending = st.number_input("Monthly Spending ($):", value=125.0)
    with sim_col2:
        sim_tickets = st.number_input("Complaint / Support Tickets:", value=3, step=1)
        sim_late = st.number_input("Late Payments Count:", value=1, step=1)
    with sim_col3:
        sim_usage = st.number_input("Usage Hours / Activity:", value=25.0)
        sim_csat = st.number_input("Satisfaction Score (1-10):", value=3, min_value=1, max_value=10, step=1)

    sim_customer = {
        "Monthly_Spending": sim_spending,
        "Support_Calls": sim_tickets,
        "Complaint_Tickets": sim_tickets,
        "Late_Payments": sim_late,
        "Total_Usage_Hours": sim_usage,
        "Satisfaction_Score": sim_csat,
    }
    sim_pred = {"risk_tier": sim_risk}

    triggered_recs = generate_retention_recommendations(sim_customer, sim_pred, industry=selected_ind)

    st.markdown("#### 🎯 Triggered Advisory Recommendations")
    for rec in triggered_recs:
        p_badge = "🔴" if rec["priority"] == "High" else ("🟡" if rec["priority"] == "Medium" else "🟢")
        st.markdown(f"{p_badge} **{rec['title']}** ({rec['priority']} Priority)")
        st.write(f"*{rec['rationale']}*")
        st.success(f"Action: {rec['suggested_action']}")

    st.markdown("---")
    st.warning("⚠️ **Academic & Enterprise Disclaimer:** Retention recommendations are predictive advisory hypotheses based on empirical behavioral patterns. They do not constitute a guaranteed churn prevention outcome and should be experimentally validated using A/B testing in your specific operational environment.")
