"""
Comprehensive End-to-End Test Suite for Churn Prediction System
Validates all requirements programmatically without modifying original data.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

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


class TestChurnSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_data_path = get_default_dataset_path()
        assert cls.raw_data_path is not None, "Raw dataset not found!"
        cls.df = load_dataset(cls.raw_data_path)

    def test_01_dataset_integrity(self):
        """Verify raw dataset is intact and has expected dimensions."""
        self.assertEqual(self.df.shape, (8000, 17))
        self.assertIn("Customer_ID", self.df.columns)
        self.assertIn("Churn", self.df.columns)
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_02_id_and_type_detection(self):
        """Verify dynamic column detection without hardcoding."""
        ids = detect_id_columns(self.df)
        self.assertIn("Customer_ID", ids)

        col_types = detect_column_types(self.df, target_col="Churn", excluded_cols=["Customer_ID"])
        self.assertNotIn("Customer_ID", col_types["numerical"])
        self.assertIn("Age", col_types["numerical"])
        self.assertIn("Subscription_Type", col_types["categorical"])

    def test_03_target_validation(self):
        """Verify validation of classification target."""
        val = validate_target_column(self.df, "Churn", churn_value=1, retained_value=0)
        self.assertTrue(val["valid"])
        self.assertEqual(len(val["errors"]), 0)
        self.assertEqual(val["class_counts"][0], 5495)
        self.assertEqual(val["class_counts"][1], 2505)

    def test_04_leak_free_preprocessing(self):
        """Verify preprocessor fits only on train and handles test transform."""
        prep = DataPreprocessor(
            target_col="Churn",
            churn_value=1,
            retained_value=0,
            excluded_columns=["Customer_ID"],
        )
        X, y = prep.prepare_data(self.df)
        self.assertEqual(len(X), 8000)
        self.assertNotIn("Customer_ID", X.columns)
        self.assertNotIn("Churn", X.columns)

        # Split simulation
        X_train, X_test = X.iloc[:6400], X.iloc[6400:]
        y_train = y.iloc[:6400]

        prep.fit(X_train, y_train)
        X_tr = prep.transform(X_train)
        X_te = prep.transform(X_test)
        self.assertEqual(X_tr.shape[0], 6400)
        self.assertEqual(X_te.shape[0], 1600)
        self.assertGreater(len(prep.get_feature_names()), 15)

    def test_05_tri_model_training_and_comparison(self):
        """Verify training of Logistic Regression, Decision Tree, Random Forest."""
        prep = DataPreprocessor(
            target_col="Churn",
            churn_value=1,
            retained_value=0,
            excluded_columns=["Customer_ID"],
        )
        trainer = ModelTrainer(test_size=0.2, selection_metric="f1")
        out = trainer.train_and_evaluate(self.df, prep)

        self.assertIn("Logistic Regression", out["results"])
        self.assertIn("Decision Tree", out["results"])
        self.assertIn("Random Forest", out["results"])

        for model_name, res in out["results"].items():
            self.assertGreaterEqual(res["accuracy"], 0.0)
            self.assertLessEqual(res["accuracy"], 1.0)
            self.assertGreaterEqual(res["f1"], 0.0)
            self.assertLessEqual(res["f1"], 1.0)
            self.assertEqual(len(res["confusion_matrix"]), 2)

        # Test bundle saving & loading
        model_save_path = os.path.join("models", "test_bundle.joblib")
        trainer.save_pipeline(model_save_path, prep)
        self.assertTrue(os.path.exists(model_save_path))

        loaded_bundle = ModelTrainer.load_pipeline(model_save_path)
        self.assertEqual(loaded_bundle["model_name"], out["best_model_name"])

        # Clean up test bundle
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_06_inference_and_attribution(self):
        """Verify real-time single and batch prediction."""
        bundle_path = os.path.join("models", "best_model.joblib")
        bundle = ModelTrainer.load_pipeline(bundle_path)

        sample = self.df.drop(columns=["Customer_ID", "Churn"]).iloc[0].to_dict()
        pred = predict_single_customer(bundle, sample)

        self.assertIn(pred["predicted_status"], ["Churned", "Retained"])
        self.assertGreaterEqual(pred["churn_probability"], 0.0)
        self.assertLessEqual(pred["churn_probability"], 1.0)
        self.assertIn(pred["risk_tier"], ["Low Risk", "Medium Risk", "High Risk"])
        self.assertGreater(len(pred["key_factors"]), 0)

        # Batch prediction
        batch_sample = self.df.drop(columns=["Customer_ID", "Churn"]).head(10)
        scored = predict_batch_customers(bundle, batch_sample)
        self.assertEqual(len(scored), 10)
        self.assertIn("Churn_Probability", scored.columns)
        self.assertIn("Risk_Tier", scored.columns)

    def test_07_analytics_visualizations(self):
        """Verify chart generation for dashboard."""
        fig1 = plot_churn_donut(self.df, "Churn", 1, 0)
        self.assertIsNotNone(fig1)

        fig2 = plot_feature_vs_churn(self.df, "Monthly_Spending", "Churn", 1, 0)
        self.assertIsNotNone(fig2)

        fig3 = plot_feature_vs_churn(self.df, "Subscription_Type", "Churn", 1, 0)
        self.assertIsNotNone(fig3)

        fig4 = plot_correlation_heatmap(self.df, ["Age", "Monthly_Spending", "Support_Calls"])
        self.assertIsNotNone(fig4)

    def test_08_retention_recommendations(self):
        """Verify industry retention suggestions."""
        sample_cust = {"Monthly_Spending": 150.0, "Support_Calls": 4, "Satisfaction_Score": 2}
        sample_pred = {"risk_tier": "High Risk"}

        for ind in INDUSTRY_PROFILES.keys():
            recs = generate_retention_recommendations(sample_cust, sample_pred, industry=ind)
            self.assertGreater(len(recs), 0)
            for r in recs:
                self.assertIn("title", r)
                self.assertIn("rationale", r)
                self.assertIn("suggested_action", r)


if __name__ == "__main__":
    unittest.main()
