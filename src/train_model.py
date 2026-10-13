"""
Machine Learning Model Training & Evaluation Module
Trains and rigorously benchmarks Logistic Regression, Decision Tree, and Random Forest.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
Ensures zero data leakage via Stratified Train-Test Splitting and fitted preprocessor isolation.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    Manages model training, evaluation, comparison, and persistence.
    """

    SUPPORTED_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        selection_metric: str = "f1",  # 'f1', 'recall', 'roc_auc', 'accuracy'
        handle_imbalance: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.selection_metric = selection_metric
        self.handle_imbalance = handle_imbalance

        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, pd.DataFrame] = {}

    def _init_models(self) -> Dict[str, Any]:
        """
        Instantiates models with balanced weights if requested to handle class imbalance.
        """
        class_weight = "balanced" if self.handle_imbalance else None
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=self.random_state,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weight,
                random_state=self.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.random_state,
            ),
        }

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end model training and evaluation:
        1. Prepares X and y.
        2. Performs stratified train-test split.
        3. Fits preprocessor ONLY on X_train.
        4. Transforms X_train and X_test.
        5. Trains Logistic Regression, Decision Tree, and Random Forest.
        6. Computes comprehensive evaluation metrics.
        7. Identifies the best-performing model based on selection_metric.
        """
        X, y = preprocessor.prepare_data(df)
        if y is None:
            raise ValueError(f"Target column '{preprocessor.target_col}' not found or invalid.")

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"Target column contains only one class ({unique_classes}). Need at least 2.")

        # Stratified split when possible
        can_stratify = len(unique_classes) == 2 and min(np.bincount(y)) >= 2
        stratify_arg = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )

        # FIT PREPROCESSOR ONLY ON TRAINING DATA
        preprocessor.fit(X_train, y_train)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        self.feature_names = preprocessor.get_feature_names()

        self.models = self._init_models()
        self.evaluation_results = {}
        self.feature_importances = {}

        best_score = -1.0
        best_name = None

        for name, model in self.models.items():
            # Train model
            model.fit(X_train_trans, y_train)

            # Predict on test set
            y_pred = model.predict(X_test_trans)

            # Predict probabilities if supported
            y_prob = None
            roc_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_trans)[:, 1]
                    roc_auc = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    roc_auc = None

            # Calculate metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Calculate feature importances
            importance_df = self._compute_feature_importance(model, self.feature_names)
            self.feature_importances[name] = importance_df

            eval_dict = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": cm,
                "classification_report": report,
                "test_size": len(y_test),
                "train_size": len(y_train),
            }
            self.evaluation_results[name] = eval_dict

            # Score tracking according to chosen selection metric
            score_to_compare = eval_dict.get(self.selection_metric, f1)
            if score_to_compare is None:
                score_to_compare = f1

            if score_to_compare > best_score:
                best_score = score_to_compare
                best_name = name

        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        return {
            "results": self.evaluation_results,
            "best_model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "test_actual": y_test.tolist(),
        }

    def _compute_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Computes standardized feature importances across linear and tree-based models.
        """
        if not feature_names:
            return pd.DataFrame(columns=["feature", "importance"])

        importances = np.zeros(len(feature_names))

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp

    def save_pipeline(
        self,
        filepath: str,
        preprocessor: DataPreprocessor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves the best trained model, preprocessor, and metadata into a joblib bundle.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "model_name": self.best_model_name,
            "model": self.best_model,
            "all_models": self.models,
            "preprocessor": preprocessor,
            "evaluation_results": self.evaluation_results,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "metadata": metadata or {},
        }
        joblib.dump(bundle, filepath)

    @staticmethod
    def load_pipeline(filepath: str) -> Dict[str, Any]:
        """
        Loads a saved pipeline bundle from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model pipeline file '{filepath}' does not exist.")
        return joblib.load(filepath)
