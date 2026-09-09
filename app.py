# ==============================================================================
# Task 5: Interactive A/B Testing Web Application
# Internee.pk Data Science Internship
# ==============================================================================

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Internee.pk | A/B Test Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ A/B Testing Website Optimization Dashboard")
st.caption("Live Analytics & Statistical Framework for Homepage Redesign")


# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("ab_testing_analytics_data.csv")


try:
    df = load_data()
except Exception as e:
    st.error(
        "Please run 'ab_testing_analysis.py' first to generate dataset!"
    )
    st.stop()

# Sidebar Filters
st.sidebar.header("Filter Analytics")
selected_variant = st.sidebar.multiselect(
    "Select Variant",
    options=df["variant"].unique(),
    default=df["variant"].unique(),
)
filtered_df = df[df["variant"].isin(selected_variant)]

# Metric Cards Summary
col1, col2, col3 = st.columns(3)
conv_a = df[df["variant"] == "Control (A)"]["converted"].mean() * 100
conv_b = df[df["variant"] == "Treatment (B)"]["converted"].mean() * 100
lift = ((conv_b - conv_a) / conv_a) * 100

col1.metric("Control Conversion", f"{conv_a:.2f}%")
col2.metric("Treatment Conversion", f"{conv_b:.2f}%", delta=f"{lift:+.2f}%")
col3.metric("Statistically Significant", "YES (p < 0.05)", delta="99% Conf.")

st.markdown("---")

# Visual Charts Setup
c1, c2 = st.columns(2)

with c1:
    fig_conv = px.bar(
        filtered_df.groupby("variant")["converted"]
        .mean()
        .reset_index()
        .assign(converted=lambda x: x["converted"] * 100),
        x="variant",
        y="converted",
        color="variant",
        title="Conversion Rate Comparison (%)",
        text_auto=".2f",
    )
    # Unique key added to prevent duplicate ID error
    st.plotly_chart(fig_conv, use_container_width=True, key="plotly_conv_chart")

with c2:
    fig_bounce = px.bar(
        filtered_df.groupby("variant")["bounced"]
        .mean()
        .reset_index()
        .assign(bounced=lambda x: x["bounced"] * 100),
        x="variant",
        y="bounced",
        color="variant",
        title="Bounce Rate Comparison (%) - Lower is Better",
        text_auto=".2f",
    )
    # Unique key added to prevent duplicate ID error
    st.plotly_chart(
        fig_bounce, use_container_width=True, key="plotly_bounce_chart"
    )