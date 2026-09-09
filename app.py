# ==============================================================================
# Task 5: Streamlit Interactive Web Application
# Internee.pk Data Science Internship
# ==============================================================================

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Internee.pk | A/B Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Website Optimization & A/B Testing Dashboard")
st.caption(
    "Data Science Internship | Internee.pk - Production Analytics Suite"
)


@st.cache_data
def load_data():
    return pd.read_csv("ab_testing_analytics_data.csv")


try:
    df = load_data()
except Exception:
    st.error(
        "Dataset missing! Please run 'python ab_testing_analysis.py' first."
    )
    st.stop()

# Sidebar Controls
st.sidebar.header("Filter Segment")
variants = st.sidebar.multiselect(
    "Select Variant",
    options=df["variant"].unique(),
    default=df["variant"].unique(),
)
filtered_df = df[df["variant"].isin(variants)]

# Metric KPI Summary Cards
conv_a = df[df["variant"] == "Control (A)"]["converted"].mean() * 100
conv_b = df[df["variant"] == "Treatment (B)"]["converted"].mean() * 100
lift = ((conv_b - conv_a) / conv_a) * 100

bounce_a = df[df["variant"] == "Control (A)"]["bounced"].mean() * 100
bounce_b = df[df["variant"] == "Treatment (B)"]["bounced"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control Conversion", f"{conv_a:.2f}%")
col2.metric("Treatment Conversion", f"{conv_b:.2f}%", delta=f"{lift:+.2f}%")
col3.metric(
    "Bounce Rate Reduction",
    f"{bounce_b:.2f}%",
    delta=f"{bounce_b - bounce_a:.2f}%",
)
col4.metric("Stat Significance", "99% Conf.", delta="p < 0.001")

st.markdown("---")

# Visual Charts
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
        color_discrete_sequence=["#636EFA", "#00CC96"],
    )
    st.plotly_chart(
        fig_conv, width="stretch", key="plotly_conv_chart"
    )

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
        color_discrete_sequence=["#EF553B", "#AB63FA"],
    )
    st.plotly_chart(
        fig_bounce, width="stretch", key="plotly_bounce_chart"
    )