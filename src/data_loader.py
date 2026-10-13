"""
Data Loader & Dataset Inspection Module
Provides generalized dataset loading (CSV/Excel), automatic type detection,
target validation, and rigorous data quality inspection.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def get_default_dataset_path() -> Optional[str]:
    """
    Locates the default raw dataset in the project directory,
    checking both 'Data/train.csv' and 'data/train.csv'.
    """
    candidates = [
        os.path.join("Data", "train.csv"),
        os.path.join("data", "train.csv"),
        os.path.join(os.path.dirname(__file__), "..", "Data", "train.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"),
    ]
    for path in candidates:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            return norm_path
    return None


def load_dataset(file_source: Any) -> pd.DataFrame:
    """
    Loads dataset from a file path, file uploader object, or buffer.
    Supports CSV and Excel files (.xlsx, .xls).
    Does NOT modify the underlying data.
    """
    if file_source is None:
        raise ValueError("No file source provided.")

    # Determine filename or name attribute if available
    filename = ""
    if isinstance(file_source, str):
        filename = file_source
    elif hasattr(file_source, "name"):
        filename = file_source.name

    filename_lower = filename.lower()

    if filename_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_source)

    # For CSV files, attempt UTF-8 and fallback to Latin-1
    try:
        if isinstance(file_source, str):
            return pd.read_csv(file_source)
        else:
            # File-like object (e.g., Streamlit UploadedFile)
            file_source.seek(0)
            return pd.read_csv(file_source)
    except UnicodeDecodeError:
        if isinstance(file_source, str):
            return pd.read_csv(file_source, encoding="latin1")
        else:
            file_source.seek(0)
            return pd.read_csv(file_source, encoding="latin1")


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically identifies candidate identifier columns such as
    Customer_ID, User_ID, Account_ID, or columns where all values are unique.
    """
    id_candidates = []
    n_rows = len(df)
    if n_rows == 0:
        return []

    for col in df.columns:
        col_lower = str(col).lower()
        # Common ID naming conventions
        name_has_id = any(term in col_lower for term in ["id", "identifier", "uuid", "guid", "code"])
        is_unique = df[col].nunique(dropna=True) == n_rows

        if name_has_id or (is_unique and not pd.api.types.is_float_dtype(df[col])):
            id_candidates.append(col)

    return id_candidates


def detect_column_types(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    excluded_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Classifies columns into numerical, categorical, datetime, and candidate IDs.
    Excludes the target column and explicitly excluded columns from features.
    """
    excluded = set(excluded_cols or [])
    if target_col:
        excluded.add(target_col)

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    candidate_id_cols = detect_id_columns(df)

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Check if it is a binary numeric column or boolean
            unique_count = series.nunique(dropna=True)
            if unique_count <= 2 and set(series.dropna().unique()).issubset({0, 1}):
                # Binary flag can be treated as numeric or categorical; numeric handles binary fine
                numerical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "candidate_ids": candidate_id_cols,
    }


def validate_target_column(
    df: pd.DataFrame,
    target_col: str,
    churn_value: Optional[Any] = None,
    retained_value: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Validates whether the selected target column is suitable for churn classification.
    Checks cardinality, missing values, class balance, and suggests mappings.
    """
    result: Dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "unique_values": [],
        "class_counts": {},
        "suggested_churn_val": None,
        "suggested_retained_val": None,
    }

    if target_col not in df.columns:
        result["errors"].append(f"Target column '{target_col}' not found in dataset.")
        return result

    target_series = df[target_col]
    null_count = target_series.isnull().sum()
    if null_count > 0:
        result["warnings"].append(
            f"Target column contains {null_count} missing values ({null_count / len(df) * 100:.2f}%). "
            "Rows with missing target values will need to be dropped during training."
        )

    valid_series = target_series.dropna()
    unique_vals = valid_series.unique().tolist()
    result["unique_values"] = unique_vals
    result["class_counts"] = valid_series.value_counts().to_dict()

    if len(unique_vals) < 2:
        result["errors"].append(
            f"Target column has only {len(unique_vals)} unique value(s). At least 2 classes are required."
        )
        return result

    if len(unique_vals) > 10:
        result["errors"].append(
            f"Target column has {len(unique_vals)} distinct values. "
            "Churn prediction requires a discrete classification label (typically binary 0/1, Yes/No)."
        )
        return result

    # Smart default value mapping detection
    val_set = set(unique_vals)
    if val_set == {0, 1} or val_set == {0.0, 1.0}:
        result["suggested_churn_val"] = 1
        result["suggested_retained_val"] = 0
    elif any(str(v).lower() in ["churn", "churned", "yes", "true", "1"] for v in unique_vals):
        for v in unique_vals:
            if str(v).lower() in ["churn", "churned", "yes", "true", "1"]:
                result["suggested_churn_val"] = v
            elif str(v).lower() in ["retained", "stayed", "no", "false", "0"]:
                result["suggested_retained_val"] = v

    # If user provided explicit values, check them
    if churn_value is not None and churn_value not in unique_vals:
        result["errors"].append(f"Specified churn value '{churn_value}' is not present in target column.")

    if retained_value is not None and retained_value not in unique_vals:
        result["errors"].append(f"Specified retained value '{retained_value}' is not present in target column.")

    if churn_value is not None and retained_value is not None and churn_value == retained_value:
        result["errors"].append("Churn value and retained value cannot be identical.")

    if not result["errors"]:
        result["valid"] = True

    return result


def inspect_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    churn_value: Optional[Any] = None,
    retained_value: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive dataset inspection and data quality audit.
    Reports facts directly from the data without fabrication.
    """
    n_rows, n_cols = df.shape

    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows * 100).round(2)
    missing_summary = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct,
    })
    columns_with_missing = missing_summary[missing_summary["Missing Count"] > 0]

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())

    # ID Candidates
    candidate_ids = detect_id_columns(df)

    # Column Types
    col_types = detect_column_types(df, target_col=target_col)

    # Data Quality Issues
    quality_issues: List[Dict[str, str]] = []

    if duplicate_rows > 0:
        quality_issues.append({
            "type": "warning",
            "message": f"Found {duplicate_rows} duplicate rows ({duplicate_rows / n_rows * 100:.2f}% of total).",
        })

    if not columns_with_missing.empty:
        worst_col = columns_with_missing["Missing %"].idxmax()
        worst_pct = columns_with_missing.loc[worst_col, "Missing %"]
        quality_issues.append({
            "type": "warning",
            "message": f"{len(columns_with_missing)} column(s) have missing values. Highest is '{worst_col}' with {worst_pct}%.",
        })

    # Constant / zero-variance columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        quality_issues.append({
            "type": "warning",
            "message": f"Columns with zero variance (single value): {', '.join(constant_cols)}. These provide no predictive signal.",
        })

    # High cardinality categorical check
    for col in col_types["categorical"]:
        unique_cnt = df[col].nunique(dropna=True)
        if unique_cnt > 50 and unique_cnt / n_rows > 0.2:
            quality_issues.append({
                "type": "warning",
                "message": f"Categorical column '{col}' has high cardinality ({unique_cnt} distinct values).",
            })

    # Target class distribution if target is present
    target_info = None
    if target_col and target_col in df.columns:
        target_info = validate_target_column(df, target_col, churn_value, retained_value)
        class_counts = target_info["class_counts"]
        total_valid = sum(class_counts.values()) if class_counts else 0
        if total_valid > 0 and len(class_counts) == 2:
            counts = list(class_counts.values())
            minority_pct = min(counts) / total_valid * 100
            if minority_pct < 20.0:
                quality_issues.append({
                    "type": "warning",
                    "message": f"Class imbalance detected: minority class represents {minority_pct:.1f}% of labeled data.",
                })

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_summary": missing_summary,
        "has_missing": not columns_with_missing.empty,
        "columns_with_missing_count": len(columns_with_missing),
        "duplicate_rows": duplicate_rows,
        "candidate_ids": candidate_ids,
        "column_types": col_types,
        "target_info": target_info,
        "quality_issues": quality_issues,
    }
