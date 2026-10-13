"""
Configurable Preprocessing Module
Builds a reusable Scikit-Learn preprocessing pipeline that dynamically adapts
to arbitrary datasets, feature sets, and target labels.
Guarantees strict isolation between training and testing data to prevent data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Generalized and configurable preprocessor for customer churn classification datasets.
    """

    def __init__(
        self,
        target_col: str = "Churn",
        churn_value: Any = 1,
        retained_value: Any = 0,
        excluded_columns: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaler: str = "standard",  # 'standard', 'minmax', 'robust', 'none'
        categorical_encoder: str = "onehot",  # 'onehot', 'ordinal'
        impute_numeric: str = "median",
        impute_categorical: str = "most_frequent",
    ):
        self.target_col = target_col
        self.churn_value = churn_value
        self.retained_value = retained_value
        self.excluded_columns = list(excluded_columns) if excluded_columns else []
        self.numerical_features = list(numerical_features) if numerical_features is not None else None
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self.scaler = scaler
        self.categorical_encoder = categorical_encoder
        self.impute_numeric = impute_numeric
        self.impute_categorical = impute_categorical

        self.pipeline: Optional[ColumnTransformer] = None
        self.fitted_numerical_cols: List[str] = []
        self.fitted_categorical_cols: List[str] = []
        self.transformed_feature_names: List[str] = []
        self.is_fitted: bool = False

    def _determine_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Determines numerical and categorical features if not explicitly provided.
        Excludes specified excluded columns.
        """
        cols_to_use = [c for c in X.columns if c not in self.excluded_columns and c != self.target_col]

        if self.numerical_features is not None and self.categorical_features is not None:
            num_cols = [c for c in self.numerical_features if c in cols_to_use]
            cat_cols = [c for c in self.categorical_features if c in cols_to_use]
            return num_cols, cat_cols

        num_cols = []
        cat_cols = []
        for col in cols_to_use:
            if pd.api.types.is_numeric_dtype(X[col]):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        return num_cols, cat_cols

    def _build_pipeline(self) -> ColumnTransformer:
        """
        Constructs the Scikit-Learn ColumnTransformer pipeline based on configuration.
        """
        transformers = []

        # Numerical pipeline
        if self.fitted_numerical_cols:
            num_steps = [("imputer", SimpleImputer(strategy=self.impute_numeric))]
            if self.scaler == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif self.scaler == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            elif self.scaler == "robust":
                num_steps.append(("scaler", RobustScaler()))
            # If 'none', no scaling step
            num_pipeline = Pipeline(steps=num_steps)
            transformers.append(("num", num_pipeline, self.fitted_numerical_cols))

        # Categorical pipeline
        if self.fitted_categorical_cols:
            cat_steps = [("imputer", SimpleImputer(strategy=self.impute_categorical, fill_value="Missing"))]
            if self.categorical_encoder == "onehot":
                cat_steps.append((
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ))
            else:
                cat_steps.append((
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ))
            cat_pipeline = Pipeline(steps=cat_steps)
            transformers.append(("cat", cat_pipeline, self.fitted_categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extracts feature matrix X and target vector y from raw dataframe.
        Maps target column values to binary {0, 1}:
            1 = Churned
            0 = Retained
        """
        working_df = df.copy()

        y = None
        if self.target_col in working_df.columns:
            target_series = working_df[self.target_col]
            # Drop rows where target is NaN
            valid_mask = target_series.notnull()
            working_df = working_df[valid_mask]
            target_series = working_df[self.target_col]

            # Map to binary 0 and 1
            # Standardize string representations if needed
            def map_val(val: Any) -> int:
                if str(val).strip().lower() == str(self.churn_value).strip().lower():
                    return 1
                if str(val).strip().lower() == str(self.retained_value).strip().lower():
                    return 0
                # Fallback for integer/float matches
                if val == self.churn_value:
                    return 1
                if val == self.retained_value:
                    return 0
                return 0

            y = target_series.apply(map_val).astype(int)

        # Drop target and excluded columns for X
        cols_to_drop = [c for c in [self.target_col] + self.excluded_columns if c in working_df.columns]
        X = working_df.drop(columns=cols_to_drop, errors="ignore")

        return X, y

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """
        Fits preprocessing steps ONLY on the training features to prevent data leakage.
        """
        self.fitted_numerical_cols, self.fitted_categorical_cols = self._determine_feature_types(X)
        self.pipeline = self._build_pipeline()

        if self.pipeline and (self.fitted_numerical_cols or self.fitted_categorical_cols):
            self.pipeline.fit(X)

            # Determine transformed feature names
            names = []
            if self.fitted_numerical_cols:
                names.extend(self.fitted_numerical_cols)
            if self.fitted_categorical_cols:
                cat_trans = self.pipeline.named_transformers_["cat"]
                encoder = cat_trans.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    cat_names = encoder.get_feature_names_out(self.fitted_categorical_cols).tolist()
                else:
                    cat_names = self.fitted_categorical_cols
                names.extend(cat_names)

            self.transformed_feature_names = names
            self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input features using the fitted pipeline.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform.")

        # Ensure all fitted columns are present in X, fill missing with NaN if absent
        X_aligned = X.copy()
        for col in self.fitted_numerical_cols + self.fitted_categorical_cols:
            if col not in X_aligned.columns:
                X_aligned[col] = np.nan

        return self.pipeline.transform(X_aligned)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fits and transforms in a single call.
        """
        return self.fit(X, y).transform(X)

    def transform_single_record(self, record_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single customer input dictionary for real-time inference.
        """
        df_single = pd.DataFrame([record_dict])
        return self.transform(df_single)

    def get_feature_names(self) -> List[str]:
        """
        Returns list of feature names after preprocessing transformation.
        """
        return self.transformed_feature_names

    def save(self, filepath: str) -> None:
        """
        Serializes the preprocessor to disk.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """
        Loads serialized preprocessor from disk.
        """
        return joblib.load(filepath)
