#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import difflib 

st.set_page_config(page_title="Raw to Ready ✨", page_icon="🧹", layout="wide")

# ---------------------------
# Custom CSS for Theme
# ---------------------------
theme_css = """
<style>
    body {
        background-color: #F4F6F6;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 30px;
    }
    .report-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        border-left: 6px solid #2E86C1;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .delta-positive { color: green; }
    .delta-negative { color: red; }
    .delta-neutral { color: black; } /* ✅ force black Δ 0 */
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def standardize_dates(series):
    def parse_date(x):
        for fmt in ("%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y", "%b %d, %Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(str(x), fmt).strftime("%Y-%m-%d")
            except:
                continue
        return x
    return series.apply(parse_date)

def normalize_text(series, col_name=""):
    """Normalize capitalization for names/cities, but skip emails."""
    if "email" in col_name.lower():
        return series  
    return series.astype(str).str.strip().str.lower().str.title()

def validate_emails(series):
    return series.apply(lambda x: x if re.match(r"[^@]+@[^@]+\.[^@]+", str(x)) else "invalid@example.com")

def fill_missing(df, method="N/A"):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            if method == "Drop Rows":
                df_copy.dropna(inplace=True)
            elif method == "N/A":
                df_copy[col].fillna("N/A", inplace=True)
            elif method == "Mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif method == "Median" and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif method == "Most Frequent":
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    return df_copy

def fuzzy_standardize(series, cutoff=0.85):
    series = series.astype(str).str.strip()
    unique_vals = series.dropna().unique()
    mapping = {}

    for val in unique_vals:
        match = difflib.get_close_matches(val, mapping.keys(), n=1, cutoff=cutoff)
        if match:
            mapping[val] = mapping[match[0]]   
        else:
            mapping[val] = val                
    return series.map(mapping)

def detect_anomalies(df, threshold=3):
    anomalies = pd.DataFrame()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:  # avoid divide by zero
            continue
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        anomaly_mask = np.abs(z_scores) > threshold
        if anomaly_mask.any():
            col_anomalies = df[anomaly_mask].copy()
            col_anomalies["Anomaly_Column"] = col
            col_anomalies["Anomaly_Value"] = df[col][anomaly_mask]
            anomalies = pd.concat([anomalies, col_anomalies])
    return anomalies

# ---------------------------
# Reset state when a new file is uploaded
# ---------------------------
def reset_cleaning_options():
    st.session_state["do_duplicates"] = False
    st.session_state["do_standardize_cols"] = False
    st.session_state["do_normalize_text"] = False
    st.session_state["do_fix_dates"] = False
    st.session_state["do_validate_emails"] = False
    st.session_state["do_fuzzy_standardize"] = False
    st.session_state["do_anomaly_detection"] = False
    st.session_state["fill_method"] = "N/A"
    st.session_state["cleaned_ready"] = False

# ---------------------------
# Hero Section
# ---------------------------
col1, col2, col3 = st.columns([1,2,1])  
with col2:
    st.image("logo.png", use_container_width=False) 
st.markdown("Welcome to Raw to Ready! ✨ This tool makes data cleaning super simple — no need to worry about messy spreadsheets. Just upload your CSV, choose what you’d like to fix, and we’ll take care of the rest. In a few clicks, you’ll have a clean, ready-to-use dataset for your projects!", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Data Cleaning Wizard")
st.sidebar.markdown("Follow the steps below:")

# Step 1: Upload
uploaded_file = st.sidebar.file_uploader("### 📥 Step 1: Upload CSV", type=["csv"])

# Reset cleaning options if a new file is uploaded
if uploaded_file is not None and "last_uploaded" not in st.session_state:
    st.session_state["last_uploaded"] = uploaded_file.name
    reset_cleaning_options()
elif uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded"):
    st.session_state["last_uploaded"] = uploaded_file.name
    reset_cleaning_options()

# ---------------------------
# If file uploaded
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Save original stats
    rows_before = int(len(df))
    nulls_before = int(df.isnull().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    # Step 2: Options
    st.sidebar.markdown("### ⚙️ Step 2: Choose Cleaning Options")
    st.sidebar.caption("Select all options that apply to your dataset. Hover over each ❓ for guidance.")
    fill_method = st.sidebar.selectbox(
        "Missing Values",
        ["Fill with N/A", "Fill with Mean", "Fill with Median", "Fill by most common", "Drop Rows"],
        key="fill_method", 
         help=("💡 Tip: For small datasets, filling values is better. If your dataset is big, you can consider dropping the rows.")
    )

    with st.sidebar.expander("Advanced Options"):
        st.checkbox("Remove duplicates", key="do_duplicates", 
                    help="Removes rows that are exact duplicates. Recommended if your dataset has repeated entries.")
        st.checkbox("Standardize column names", key="do_standardize_cols", 
                    help="Converts column names to lowercase and replaces spaces with underscores for consistency.")
        st.checkbox("Normalize text", key="do_normalize_text", 
                    help="Makes text consistent (e.g., 'new york' → 'New York'). Skips emails automatically.")
        st.checkbox("Fix date formats", key="do_fix_dates", 
                    help="Converts different date styles (e.g., '01/02/23', 'Feb 1, 2023') into YYYY-MM-DD format.")
        st.checkbox("Validate emails", key="do_validate_emails",
                    help="Ensures all emails follow a proper format. Invalid ones become `invalid@example.com`.")
        st.checkbox("Fuzzy standardize values", key="do_fuzzy_standardize", 
                    help="Groups similar text values together (e.g., 'NYC', 'New York City', 'N.Y.C.' → 'NYC').")
        st.checkbox("Detect anomalies", key="do_anomaly_detection", 
                    help="Flags unusual numeric values using statistical detection. Useful for spotting outliers (extreme values).")

    # Tabs for Raw vs Cleaned data
    tab1, tab2, tab3 = st.tabs(["🧹 Raw Data Preview", "✨ Cleaned Data Preview", "⚠️ Anomalies Detected"])
    with tab1:
        st.dataframe(df.head(10))

    # Step 3: Run Cleaning
    st.sidebar.markdown("#### 🧹 Step 3: Apply Cleaning")
    if st.sidebar.button("Run Cleaning"):
        df_cleaned = df.copy()

        # Progress bar
        progress = st.progress(0)

        # Cleaning steps
        progress.progress(10)
        df_cleaned = fill_missing(df_cleaned, method=fill_method)

        progress.progress(30)
        if st.session_state["do_duplicates"]:
            df_cleaned.drop_duplicates(inplace=True)

        progress.progress(50)
        if st.session_state["do_standardize_cols"]:
            df_cleaned.columns = [c.strip().lower().replace(" ", "_") for c in df_cleaned.columns]

        progress.progress(70)
        if st.session_state["do_normalize_text"]:
            for col in df_cleaned.select_dtypes(include=["object"]).columns:
                df_cleaned[col] = normalize_text(df_cleaned[col], col_name=col)

        progress.progress(80)
        if st.session_state["do_fix_dates"]:
            for col in df_cleaned.columns:
                if "date" in col.lower():
                    df_cleaned[col] = standardize_dates(df_cleaned[col])
        if st.session_state["do_validate_emails"]:
            for col in df_cleaned.columns:
                if "email" in col.lower():
                    df_cleaned[col] = validate_emails(df_cleaned[col])

        progress.progress(90)
        if st.session_state["do_fuzzy_standardize"]:
            for col in df_cleaned.select_dtypes(include=["object"]).columns:
                df_cleaned[col] = fuzzy_standardize(df_cleaned[col], cutoff=0.85)

        progress.progress(95)
        anomalies = pd.DataFrame()
        if st.session_state["do_anomaly_detection"]:
          anomalies = detect_anomalies(df_cleaned)

        progress.progress(100)

        with tab2:
            st.dataframe(df_cleaned.head(10))
        
        with tab3:
            if "anomalies" in locals() and not anomalies.empty:
                st.warning(f"{len(anomalies)} anomalies detected ⚠️")
                st.dataframe(anomalies)
            else:
                st.success("No anomalies detected ✅")


        # Save cleaned stats
        rows_after = int(len(df_cleaned))
        nulls_after = int(df_cleaned.isnull().sum().sum())
        duplicates_after = int(df_cleaned.duplicated().sum())

                              # Report
        st.success("Cleaning completed successfully!")

        col1, col2, col3 = st.columns(3)
        with col1:
            delta = rows_after - rows_before
            color = "black" if delta == 0 else ("green" if delta > 0 else "red")
            st.markdown(
                f"""
                <div class='report-card'>
                    <h3>Rows</h3>
                    <p style='font-size:26px; font-weight:bold;'>{rows_after}</p>
                    <div style='font-size:28px; font-weight:bold; color:{color};'>
                        Δ {delta}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            delta = nulls_before - nulls_after
            color = "black" if delta == 0 else "green"
            st.markdown(
                f"""
                <div class='report-card'>
                    <h3>Nulls</h3>
                    <p style='font-size:26px; font-weight:bold;'>{nulls_after}</p>
                    <div style='font-size:28px; font-weight:bold; color:{color};'>
                        Fixed {delta}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            delta = duplicates_before - duplicates_after
            color = "black" if delta == 0 else "green"
            st.markdown(
                f"""
                <div class='report-card'>
                    <h3>Duplicates</h3>
                    <p style='font-size:26px; font-weight:bold;'>{duplicates_after}</p>
                    <div style='font-size:28px; font-weight:bold; color:{color};'>
                        Removed {delta}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Step 4: Download
        st.subheader("📥 Step 4: Save")
        csv = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info(" Upload a CSV file in the sidebar to get started!")
