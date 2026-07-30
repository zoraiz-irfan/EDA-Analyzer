"""
All chart-producing functions. Uses Plotly so charts are interactive
in Streamlit (zoom/hover) instead of static matplotlib PNGs.
"""

import streamlit as st
import plotly.express as px
import pandas as pd


def plot_numeric_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns found.")
        return

    col_choice = st.selectbox("Choose a numeric column", numeric_cols, key="hist_col")
    fig = px.histogram(df, x=col_choice, marginal="box", title=f"Distribution of {col_choice}")
    st.plotly_chart(fig, use_container_width=True)


def plot_categorical_counts(df: pd.DataFrame, max_unique: int = 30):
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique() <= max_unique]
    if not cat_cols:
        st.info("No categorical columns with a reasonable number of categories found.")
        return

    col_choice = st.selectbox("Choose a categorical column", cat_cols, key="cat_col")
    counts = df[col_choice].value_counts().reset_index()
    counts.columns = [col_choice, "Count"]
    fig = px.bar(counts, x=col_choice, y="Count", title=f"Value counts for {col_choice}")
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    if corr_matrix.empty:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
        return
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_scatter_bivariate(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for a scatter plot.")
        return

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X-axis", numeric_cols, index=0, key="scatter_x")
    y_col = c2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="scatter_color")
    color_arg = None if color_col == "None" else color_col

    fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=f"{x_col} vs {y_col}")
    st.plotly_chart(fig, use_container_width=True)


def plot_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns found.")
        return
    col_choice = st.selectbox("Choose a numeric column", numeric_cols, key="box_col")
    fig = px.box(df, y=col_choice, title=f"Boxplot of {col_choice}")
    st.plotly_chart(fig, use_container_width=True)


def plot_pairwise(df: pd.DataFrame, max_cols: int = 5):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for a pairwise plot.")
        return

    default_cols = numeric_cols[:min(4, len(numeric_cols))]
    chosen = st.multiselect("Columns to include (max 5 recommended)", numeric_cols, default=default_cols)
    if len(chosen) < 2:
        st.warning("Select at least 2 columns.")
        return
    if len(chosen) > max_cols:
        st.warning(f"Showing first {max_cols} selected columns to keep this readable.")
        chosen = chosen[:max_cols]

    fig = px.scatter_matrix(df[chosen], dimensions=chosen, title="Pairwise Relationships")
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)
