import streamlit as st
from utils.file_loader import load_file
from utils.eda_core import (
    show_overview, show_missing,
    show_univariate, show_correlation,
    show_outliers, show_cleaning_panel,
    show_ml_recommender
)

st.set_page_config(
    page_title="AutoEDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AutoEDA — Automated Exploratory Data Analysis")
st.markdown("Upload any file and get instant EDA. Supports **CSV, Excel, JSON, XML**.")

with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls", "json", "xml"]
    )

    st.markdown("---")
    st.header("Analysis Sections")
    sections = {
        "Overview": st.checkbox("Overview", value=True),
        "Missing Values": st.checkbox("Missing Values", value=True),
        "Univariate": st.checkbox("Univariate Analysis", value=True),
        "Correlation": st.checkbox("Correlation", value=True),
        "Outliers": st.checkbox("Outliers", value=True),
        "Cleaning": st.checkbox("Data Cleaning", value=True),
        "ML Recommender": st.checkbox("ML Recommender", value=True),
    }

    st.markdown("---")
    st.markdown("Built with Streamlit")

if uploaded_file is not None:
    df = load_file(uploaded_file)

    if df is not None:
        st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows x {df.shape[1]} columns")
        st.markdown("---")

        if sections["Overview"]:
            show_overview(df)
            st.markdown("---")

        if sections["Missing Values"]:
            show_missing(df)
            st.markdown("---")

        if sections["Univariate"]:
            show_univariate(df)
            st.markdown("---")

        if sections["Correlation"]:
            show_correlation(df)
            st.markdown("---")

        if sections["Outliers"]:
            show_outliers(df)
            st.markdown("---")

        if sections["Cleaning"]:
            show_cleaning_panel(df)
            st.markdown("---")

        if sections["ML Recommender"]:
            show_ml_recommender(df)
            st.markdown("---")

        st.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a file from the sidebar to begin.")
