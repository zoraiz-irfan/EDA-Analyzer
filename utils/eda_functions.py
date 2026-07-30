"""
Core EDA logic — all functions take a pandas DataFrame and return
either a DataFrame/dict/string that app.py can render with Streamlit.
Keeping this separate from app.py means the analysis logic has
no dependency on Streamlit and can be unit-tested or reused.
"""

import pandas as pd
import numpy as np


def get_overview(df: pd.DataFrame) -> dict:
    """Basic shape, dtypes, memory usage."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include="datetime").columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count + percentage of missing values per column, sorted worst-first."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    result = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.values,
    })
    result = result[result["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    return result.reset_index(drop=True)


def get_duplicates(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / len(df) * 100, 2) if len(df) else 0,
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """describe() for numeric AND categorical columns combined."""
    return df.describe(include="all").transpose()


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def get_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Flag column pairs with |correlation| above threshold (excluding self-pairs)."""
    if corr_matrix.empty:
        return pd.DataFrame()

    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({"Feature 1": cols[i], "Feature 2": cols[j], "Correlation": round(val, 3)})
    return pd.DataFrame(pairs).sort_values("Correlation", ascending=False) if pairs else pd.DataFrame()


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-method outlier detection for every numeric column."""
    numeric_df = df.select_dtypes(include="number")
    rows = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append({
            "Column": col,
            "Outlier Count": len(outliers),
            "Outlier %": round(len(outliers) / len(series) * 100, 2),
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Outlier Count", ascending=False).reset_index(drop=True) if not result.empty else result


def get_categorical_summary(df: pd.DataFrame, max_unique: int = 20) -> dict:
    """Value counts for each categorical column (skips high-cardinality columns)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    summary = {}
    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique <= max_unique:
            summary[col] = df[col].value_counts().head(max_unique)
    return summary


def get_skewness_kurtosis(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    rows = []
    for col in numeric_df.columns:
        rows.append({
            "Column": col,
            "Skewness": round(numeric_df[col].skew(), 3),
            "Kurtosis": round(numeric_df[col].kurt(), 3),
        })
    return pd.DataFrame(rows)


def generate_feature_suggestions(df: pd.DataFrame) -> list:
    """Simple rule-based suggestions — cheap sanity checks before the AI summary."""
    suggestions = []
    missing = get_missing_values(df)
    dup = get_duplicates(df)
    outliers = detect_outliers_iqr(df)
    skew = get_skewness_kurtosis(df)

    if not missing.empty:
        worst = missing.iloc[0]
        suggestions.append(
            f"Column '{worst['Column']}' has {worst['Missing %']}% missing values — "
            f"consider imputation or dropping it depending on importance."
        )

    if dup["duplicate_rows"] > 0:
        suggestions.append(f"Found {dup['duplicate_rows']} duplicate rows ({dup['duplicate_pct']}%) — consider dropping them.")

    if not outliers.empty:
        top = outliers.iloc[0]
        if top["Outlier Count"] > 0:
            suggestions.append(
                f"Column '{top['Column']}' has {top['Outlier Count']} potential outliers — "
                f"consider capping, transforming, or investigating these rows."
            )

    if not skew.empty:
        skewed_cols = skew[skew["Skewness"].abs() > 1]
        if not skewed_cols.empty:
            names = ", ".join(skewed_cols["Column"].tolist()[:5])
            suggestions.append(f"Highly skewed columns detected ({names}) — consider a log or power transform.")

    high_corr = get_high_correlations(get_correlation_matrix(df))
    if not high_corr.empty:
        pair = high_corr.iloc[0]
        suggestions.append(
            f"'{pair['Feature 1']}' and '{pair['Feature 2']}' are highly correlated "
            f"({pair['Correlation']}) — consider dropping one to reduce multicollinearity."
        )

    if not suggestions:
        suggestions.append("No major data quality issues detected — dataset looks clean.")

    return suggestions


def build_dataset_context_string(df: pd.DataFrame, max_chars: int = 3000) -> str:
    """
    Compact textual summary of the dataset, used as context fed to the
    Groq LLM (so the model can 'see' the data without receiving the full file).
    """
    overview = get_overview(df)
    missing = get_missing_values(df)
    dup = get_duplicates(df)
    stats = df.describe(include="all").transpose().round(3)

    parts = [
        f"Rows: {overview['rows']}, Columns: {overview['columns']}",
        f"Numeric columns: {overview['numeric_columns']}",
        f"Categorical columns: {overview['categorical_columns']}",
        f"Duplicate rows: {dup['duplicate_rows']} ({dup['duplicate_pct']}%)",
        f"Missing values summary:\n{missing.to_string(index=False) if not missing.empty else 'None'}",
        f"Descriptive statistics:\n{stats.to_string()}",
    ]
    text = "\n\n".join(parts)
    return text[:max_chars]
