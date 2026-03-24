import pandas as pd
import streamlit as st

SUPPORTED_FORMATS = ["csv", "xlsx", "xls", "json", "xml"]

def load_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext not in SUPPORTED_FORMATS:
        st.error(f"Unsupported format: .{ext}. Please upload CSV, Excel, JSON, or XML.")
        return None

    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)

        elif ext in ["xlsx", "xls"]:
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            if len(sheet_names) > 1:
                selected = st.selectbox("Multiple sheets found. Select one:", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected)
            else:
                df = pd.read_excel(uploaded_file)

        elif ext == "json":
            df = pd.read_json(uploaded_file)

        elif ext == "xml":
            df = pd.read_xml(uploaded_file)

        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
