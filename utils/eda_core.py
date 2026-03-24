import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st
import plotly.express as px

# ── 1. Dataset Overview ──────────────────────────────────────────────
def show_overview(df):
    st.subheader(" Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())
    col4.metric("Missing Cells", df.isnull().sum().sum())

    st.markdown("#### Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
        "Unique Values": df.nunique().values
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("#### Sample Data (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)


# ── 2. Missing Values ─────────────────────────────────────────────────
def show_missing(df):
    st.subheader(" Missing Value Analysis")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found in the dataset!")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Missing Value Counts**")
        miss_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        }).sort_values("Missing %", ascending=False)
        st.dataframe(miss_df, use_container_width=True)

    with col2:
        st.markdown("**Missing Value Heatmap**")
        fig, ax = plt.subplots(figsize=(8, 4))
        msno.matrix(df, ax=ax, sparkline=False)
        st.pyplot(fig)
        plt.close()


# ── 3. Univariate Analysis ────────────────────────────────────────────
def show_univariate(df):
    st.subheader("Univariate Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric
    if numeric_cols:
        st.markdown("#### Numeric Columns — Distribution")
        st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)

        selected_num = st.selectbox("Select a numeric column to visualize:", numeric_cols)
        fig = px.histogram(df, x=selected_num, marginal="box",
                           title=f"Distribution of {selected_num}",
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Categorical
    if cat_cols:
        st.markdown("#### Categorical Columns — Value Counts")
        selected_cat = st.selectbox("Select a categorical column:", cat_cols)
        top_vals = df[selected_cat].value_counts().head(20).reset_index()
        top_vals.columns = [selected_cat, "Count"]
        fig = px.bar(top_vals, x=selected_cat, y="Count",
                     title=f"Top 20 values in '{selected_cat}'",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ── 4. Correlation Analysis ───────────────────────────────────────────
def show_correlation(df):
    st.subheader("Correlation Analysis")
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    method = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True)
    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title(f"{method.capitalize()} Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    # Top correlated pairs
    st.markdown("#### Top Correlated Pairs")
    corr_pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                  .stack()
                  .reset_index())
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    corr_pairs = corr_pairs.sort_values("Abs Correlation", ascending=False).head(10)
    st.dataframe(corr_pairs.drop("Abs Correlation", axis=1).round(4),
                 use_container_width=True)


# ── 5. Outlier Detection ──────────────────────────────────────────────
def show_outliers(df):
    st.subheader("Outlier Detection (IQR Method)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found.")
        return

    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary.append({
            "Column": col, "Q1": round(Q1, 3), "Q3": round(Q3, 3),
            "IQR": round(IQR, 3), "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3), "Outlier Count": n_outliers,
            "Outlier %": round(n_outliers / len(df) * 100, 2)
        })

    st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)

    selected = st.selectbox("Select column for box plot:", numeric_cols)
    fig = px.box(df, y=selected, title=f"Box Plot — {selected}",
                 template="plotly_white", points="outliers")
    st.plotly_chart(fig, use_container_width=True)
