# ==============================================================================
# Task 5: A/B Testing & Website Optimization Engine
# Internee.pk Data Science Internship
# Tech Stack: Python, Pandas, SciPy, Statsmodels, Plotly
# ==============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# ------------------------------------------------------------------------------
# STEP 1: Synthetic Dataset Generation (Simulating Web Analytics / GA4)
# ------------------------------------------------------------------------------
def generate_ab_test_data(samples_per_group=2500, random_state=42):
    np.random.seed(random_state)

    # Variant A (Control - Current Homepage)
    # Conversion Rate ~ 10.5%, Bounce Rate ~ 45%, Session Duration ~ 120s
    a_converted = np.random.binomial(1, 0.105, samples_per_group)
    a_bounced = np.random.binomial(1, 0.45, samples_per_group)
    a_duration = np.random.normal(120, 35, samples_per_group).clip(15, 300)

    # Variant B (Treatment - Redesigned Interactive Homepage)
    # Conversion Rate ~ 13.2%, Bounce Rate ~ 38%, Session Duration ~ 145s
    b_converted = np.random.binomial(1, 0.132, samples_per_group)
    b_bounced = np.random.binomial(1, 0.38, samples_per_group)
    b_duration = np.random.normal(145, 40, samples_per_group).clip(15, 300)

    df_a = pd.DataFrame(
        {
            "user_id": [f"USR_A_{i:04d}" for i in range(samples_per_group)],
            "variant": "Control (A)",
            "converted": a_converted,
            "bounced": a_bounced,
            "session_duration_sec": np.round(a_duration, 2),
        }
    )

    df_b = pd.DataFrame(
        {
            "user_id": [f"USR_B_{i:04d}" for i in range(samples_per_group)],
            "variant": "Treatment (B)",
            "converted": b_converted,
            "bounced": b_bounced,
            "session_duration_sec": np.round(b_duration, 2),
        }
    )

    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_csv("ab_testing_analytics_data.csv", index=False)
    print("Dataset saved to 'ab_testing_analytics_data.csv'")
    return df


# ------------------------------------------------------------------------------
# STEP 2: Statistical Hypothesis Testing (Z-Test & Two-Sample t-Test)
# ------------------------------------------------------------------------------
def perform_hypothesis_tests(df):
    print("\n" + "=" * 50)
    print("      STATISTICAL HYPOTHESIS TESTING RESULTS      ")
    print("=" * 50)

    group_a = df[df["variant"] == "Control (A)"]
    group_b = df[df["variant"] == "Treatment (B)"]

    # 1. Conversion Rate Z-Test
    count_conv = [group_a["converted"].sum(), group_b["converted"].sum()]
    nobs_conv = [len(group_a), len(group_b)]
    z_stat_conv, p_val_conv = proportions_ztest(count_conv, nobs_conv)

    conv_a = group_a["converted"].mean() * 100
    conv_b = group_b["converted"].mean() * 100
    lift_conv = ((conv_b - conv_a) / conv_a) * 100

    print(
        f"\n[Conversion Rate] Control: {conv_a:.2f}% | Treatment: {conv_b:.2f}%"
    )
    print(f"Conversion Lift: {lift_conv:+.2f}%")
    print(f"Z-Statistic: {z_stat_conv:.4f} | P-Value: {p_val_conv:.4e}")
    print(
        "Result: "
        + (
            "STATISTICALLY SIGNIFICANT (Reject H0)"
            if p_val_conv < 0.05
            else "NOT SIGNIFICANT"
        )
    )

    # 2. Bounce Rate Z-Test
    count_bounce = [group_a["bounced"].sum(), group_b["bounced"].sum()]
    z_stat_bounce, p_val_bounce = proportions_ztest(count_bounce, nobs_conv)

    bounce_a = group_a["bounced"].mean() * 100
    bounce_b = group_b["bounced"].mean() * 100
    drop_bounce = ((bounce_a - bounce_b) / bounce_a) * 100

    print(f"\n[Bounce Rate] Control: {bounce_a:.2f}% | Treatment: {bounce_b:.2f}%")
    print(f"Bounce Reduction: {drop_bounce:.2f}% decrease")
    print(f"Z-Statistic: {z_stat_bounce:.4f} | P-Value: {p_val_bounce:.4e}")

    # 3. Session Duration Welch's t-Test
    t_stat_dur, p_val_dur = stats.ttest_ind(
        group_a["session_duration_sec"],
        group_b["session_duration_sec"],
        equal_var=False,
    )
    dur_a = group_a["session_duration_sec"].mean()
    dur_b = group_b["session_duration_sec"].mean()

    print(f"\n[Avg Session Duration] Control: {dur_a:.1f}s | Treatment: {dur_b:.1f}s")
    print(f"T-Statistic: {t_stat_dur:.4f} | P-Value: {p_val_dur:.4e}")

    return {
        "conv_a": conv_a,
        "conv_b": conv_b,
        "p_val_conv": p_val_conv,
        "bounce_a": bounce_a,
        "bounce_b": bounce_b,
        "dur_a": dur_a,
        "dur_b": dur_b,
    }


# ------------------------------------------------------------------------------
# STEP 3: Automated Visualization & Dashboard Export (Fixed Rendering)
# ------------------------------------------------------------------------------
def generate_and_save_visuals(df, metrics):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Metric 1: Conversion Rate
    sns.barplot(
        data=df,
        x="variant",
        y="converted",
        ax=axes[0],
        palette=["#7f7f7f", "#2ca02c"],
        hue="variant",
        legend=False,
    )
    axes[0].set_title(
        f"1. Conversion Rate (%)\nP-Value = {metrics['p_val_conv']:.2e} (Sig)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Conversion Ratio")

    # Metric 2: Bounce Rate
    sns.barplot(
        data=df,
        x="variant",
        y="bounced",
        ax=axes[1],
        palette=["#7f7f7f", "#d62728"],
        hue="variant",
        legend=False,
    )
    axes[1].set_title(
        "2. Bounce Rate (%)\n(Lower is Better)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Bounce Ratio")

    # Metric 3: Session Duration Distribution
    sns.kdeplot(
        data=df,
        x="session_duration_sec",
        hue="variant",
        fill=True,
        ax=axes[2],
        palette=["#7f7f7f", "#1f77b4"],
    )
    axes[2].set_title(
        "3. User Engagement: Session Duration (sec)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel("Seconds")

    plt.tight_layout()

    # Save PNG clean image before clearing memory
    output_png = "ab_test_results_dashboard.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"\nDashboard exported cleanly as '{output_png}'.")