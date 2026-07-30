import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from utils import eda_functions as eda
from utils import visualization as viz
from utils import groq_helper

load_dotenv()

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI EDA Analyzer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Session State Initialization
# ----------------------------
# This is the fix for your original bug: without session_state, the
# dataframe disappears every time you click a different sidebar option,
# because Streamlit reruns the whole script top-to-bottom on every interaction.
if "df" not in st.session_state:
    st.session_state.df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

# ----------------------------
# Title
# ----------------------------
st.title("📊 AI EDA Analyzer")
st.markdown(
    """
    Upload your dataset and let AI perform Exploratory Data Analysis automatically.
    """
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Select Section",
    ["Home", "Upload Dataset", "EDA", "Visualization", "AI Chat", "Report"],
)

st.sidebar.divider()
st.sidebar.subheader("🔑 Groq API Key")
api_key_input = st.sidebar.text_input(
    "Enter your Groq API key",
    type="password",
    value=os.getenv("GROQ_API_KEY", ""),
    help="Get a free key at https://console.groq.com/keys",
)

if st.session_state.df is not None:
    st.sidebar.divider()
    st.sidebar.success(f"Loaded: {st.session_state.file_name}")
    st.sidebar.write(f"Shape: {st.session_state.df.shape}")

st.sidebar.divider()
st.sidebar.info("AI EDA Analyzer v1.0")

# ----------------------------
# Home Page
# ----------------------------
if option == "Home":
    st.header("Welcome")
    st.success("Upload any CSV or Excel dataset to begin.")
    st.write(
        """
        This application automatically performs:

        ✔ Data Overview
        ✔ Missing Value Analysis
        ✔ Duplicate Analysis
        ✔ Descriptive Statistics
        ✔ Correlation Analysis
        ✔ Outlier Detection
        ✔ Skewness / Kurtosis Check
        ✔ Feature Suggestions
        ✔ Univariate / Bivariate / Multivariate Visualizations
        ✔ AI-powered Insights & Chat (via Groq)

        👉 Start from **Upload Dataset** in the sidebar.
        """
    )

# ----------------------------
# Upload Dataset
# ----------------------------
elif option == "Upload Dataset":
    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Persist to session_state so other tabs can see it
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            # Reset stale AI state from any previous dataset
            st.session_state.chat_history = []
            st.session_state.ai_summary = None

            st.success("Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        st.subheader("Columns")
        st.write(df.columns.tolist())

# ----------------------------
# EDA Page
# ----------------------------
elif option == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first from **Upload Dataset**.")
    else:
        df = st.session_state.df

        # ---- Overview ----
        st.subheader("1️⃣ Dataset Overview")
        overview = eda.get_overview(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", overview["rows"])
        c2.metric("Columns", overview["columns"])
        c3.metric("Memory (MB)", overview["memory_usage_mb"])
        st.write("**Column data types:**")
        st.dataframe(pd.DataFrame(overview["dtypes"].items(), columns=["Column", "Dtype"]))

        # ---- Missing Values ----
        st.subheader("2️⃣ Missing Values")
        missing_df = eda.get_missing_values(df)
        if missing_df.empty:
            st.success("No missing values found! 🎉")
        else:
            st.dataframe(missing_df, use_container_width=True)
            st.bar_chart(missing_df.set_index("Column")["Missing %"])

        # ---- Duplicates ----
        st.subheader("3️⃣ Duplicate Rows")
        dup = eda.get_duplicates(df)
        c1, c2 = st.columns(2)
        c1.metric("Duplicate Rows", dup["duplicate_rows"])
        c2.metric("Duplicate %", f"{dup['duplicate_pct']}%")

        # ---- Descriptive Stats ----
        st.subheader("4️⃣ Descriptive Statistics")
        st.dataframe(eda.get_descriptive_stats(df), use_container_width=True)

        # ---- Skewness / Kurtosis ----
        st.subheader("5️⃣ Skewness & Kurtosis")
        skew_df = eda.get_skewness_kurtosis(df)
        if skew_df.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(skew_df, use_container_width=True)

        # ---- Correlation ----
        st.subheader("6️⃣ Correlation Analysis")
        corr_matrix = eda.get_correlation_matrix(df)
        if corr_matrix.empty:
            st.info("Need at least 2 numeric columns for correlation analysis.")
        else:
            viz.plot_correlation_heatmap(corr_matrix)
            high_corr = eda.get_high_correlations(corr_matrix)
            if not high_corr.empty:
                st.write("**Highly correlated pairs (|r| ≥ 0.75):**")
                st.dataframe(high_corr, use_container_width=True)

        # ---- Outliers ----
        st.subheader("7️⃣ Outlier Detection (IQR method)")
        outliers_df = eda.detect_outliers_iqr(df)
        if outliers_df.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(outliers_df, use_container_width=True)

        # ---- Categorical Summary ----
        st.subheader("8️⃣ Categorical Column Summary")
        cat_summary = eda.get_categorical_summary(df)
        if not cat_summary:
            st.info("No low-cardinality categorical columns found.")
        else:
            for col, counts in cat_summary.items():
                with st.expander(f"Column: {col}"):
                    st.bar_chart(counts)

        # ---- Feature Suggestions ----
        st.subheader("9️⃣ Feature Suggestions")
        suggestions = eda.generate_feature_suggestions(df)
        for s in suggestions:
            st.write(f"- {s}")

# ----------------------------
# Visualization Page
# ----------------------------
elif option == "Visualization":
    st.header("📈 Visualizations")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first from **Upload Dataset**.")
    else:
        df = st.session_state.df

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Univariate", "Categorical", "Bivariate", "Boxplot", "Multivariate"]
        )

        with tab1:
            viz.plot_numeric_distributions(df)
        with tab2:
            viz.plot_categorical_counts(df)
        with tab3:
            viz.plot_scatter_bivariate(df)
        with tab4:
            viz.plot_boxplots(df)
        with tab5:
            viz.plot_pairwise(df)

# ----------------------------
# AI Chat Page
# ----------------------------
elif option == "AI Chat":
    st.header("🤖 AI Chat & Insights (Groq)")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first from **Upload Dataset**.")
    elif not api_key_input:
        st.warning("Please enter your Groq API key in the sidebar to use this feature.")
    else:
        df = st.session_state.df
        dataset_context = eda.build_dataset_context_string(df)

        st.subheader("Automatic AI Insight Summary")
        if st.button("Generate AI Summary") or st.session_state.ai_summary:
            if st.session_state.ai_summary is None:
                with st.spinner("Analyzing dataset with Groq..."):
                    try:
                        st.session_state.ai_summary = groq_helper.generate_ai_summary(
                            api_key_input, dataset_context
                        )
                    except Exception as e:
                        st.error(f"Groq API error: {e}")
            if st.session_state.ai_summary:
                st.markdown(st.session_state.ai_summary)

        st.divider()
        st.subheader("Ask Questions About Your Data")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Ask something about your dataset...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = groq_helper.chat_about_dataset(
                            api_key_input,
                            dataset_context,
                            st.session_state.chat_history[:-1],  # history before this turn
                            user_question,
                        )
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

# ----------------------------
# Report Page
# ----------------------------
elif option == "Report":
    st.header("📄 EDA Report")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first from **Upload Dataset**.")
    else:
        df = st.session_state.df
        overview = eda.get_overview(df)
        missing_df = eda.get_missing_values(df)
        dup = eda.get_duplicates(df)
        outliers_df = eda.detect_outliers_iqr(df)
        suggestions = eda.generate_feature_suggestions(df)

        report_lines = [
            f"# EDA Report — {st.session_state.file_name}",
            f"\n**Rows:** {overview['rows']}  \n**Columns:** {overview['columns']}",
            f"\n**Duplicate rows:** {dup['duplicate_rows']} ({dup['duplicate_pct']}%)",
            "\n## Missing Values",
            missing_df.to_markdown(index=False) if not missing_df.empty else "No missing values.",
            "\n## Outliers (IQR method)",
            outliers_df.to_markdown(index=False) if not outliers_df.empty else "No numeric columns.",
            "\n## Feature Suggestions",
            "\n".join(f"- {s}" for s in suggestions),
        ]

        if st.session_state.ai_summary:
            report_lines.append("\n## AI-Generated Insights")
            report_lines.append(st.session_state.ai_summary)

        report_text = "\n".join(report_lines)
        st.markdown(report_text)

        st.download_button(
            "📥 Download Report (Markdown)",
            data=report_text,
            file_name=f"eda_report_{st.session_state.file_name}.md",
            mime="text/markdown",
        )
