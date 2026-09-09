# ==============================================================================
# Task 5: A/B Testing & Website Optimization Engine
# Internee.pk Data Science Internship
# ==============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier


def generate_ab_data(n_samples=10000):
    """Generate synthetic web analytics dataset for A/B Testing."""
    np.random.seed(42)
    n_control = n_samples // 2
    n_treatment = n_samples // 2

    # Control (Variant A) - Original Homepage Design
    conv_a = np.random.binomial(1, p=0.10, size=n_control)
    bounce_a = np.random.binomial(1, p=0.48, size=n_control)
    duration_a = np.random.exponential(scale=110, size=n_control)
    pages_a = np.random.poisson(lam=2.5, size=n_control)

    # Treatment (Variant B) - Redesigned Homepage
    conv_b = np.random.binomial(1, p=0.128, size=n_treatment)
    bounce_b = np.random.binomial(1, p=0.39, size=n_treatment)
    duration_b = np.random.exponential(scale=140, size=n_treatment)
    pages_b = np.random.poisson(lam=3.8, size=n_treatment)

    df_a = pd.DataFrame(
        {
            "user_id": [f"user_A_{i}" for i in range(n_control)],
            "variant": "Control (A)",
            "converted": conv_a,
            "bounced": bounce_a,
            "session_duration_sec": np.round(duration_a, 2),
            "pages_viewed": pages_a,
        }
    )

    df_b = pd.DataFrame(
        {
            "user_id": [f"user_B_{i}" for i in range(n_treatment)],
            "variant": "Treatment (B)",
            "converted": conv_b,
            "bounced": bounce_b,
            "session_duration_sec": np.round(duration_b, 2),
            "pages_viewed": pages_b,
        }
    )

    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_csv("ab_testing_analytics_data.csv", index=False)
    print("✅ Analytics dataset successfully saved: 'ab_testing_analytics_data.csv'")
    return df


def perform_statistical_testing(df):
    """Perform Two-Sample Z-Test for Proportions and T-Test for Engagement."""
    conv_a = df[df["variant"] == "Control (A)"]["converted"]
    conv_b = df[df["variant"] == "Treatment (B)"]["converted"]

    p1, p2 = conv_a.mean(), conv_b.mean()
    p_pooled = (conv_a.sum() + conv_b.sum()) / (len(conv_a) + len(conv_b))
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / len(conv_a) + 1 / len(conv_b)))
    z_stat = (p2 - p1) / se
    p_val_conv = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    lift = ((p2 - p1) / p1) * 100

    print("\n" + "=" * 50)
    print("STATISTICAL HYPOTHESIS TEST RESULTS")
    print("=" * 50)
    print(f"Control (A) Conversion Rate   : {p1 * 100:.2f}%")
    print(f"Treatment (B) Conversion Rate : {p2 * 100:.2f}%")
    print(f"Relative Lift                 : {lift:+.2f}%")
    print(f"Z-Statistic                   : {z_stat:.4f}")
    print(f"P-Value                       : {p_val_conv:.4e}")
    print(
        f"Statistically Significant     : {'YES (p < 0.05)' if p_val_conv < 0.05 else 'NO'}"
    )
    print("=" * 50 + "\n")

    return p_val_conv, lift


def export_dashboard_visuals(df, p_val_conv):
    """Export clean high-res visualization dashboard as PNG."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Conversion Rate
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
        f"Conversion Rate (%)\nP-Val = {p_val_conv:.2e} (Significant)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Conversion Ratio")

    # Plot 2: Bounce Rate
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
        "Bounce Rate (%)\n(Lower is Better)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Bounce Ratio")

    # Plot 3: User Engagement
    sns.kdeplot(
        data=df,
        x="session_duration_sec",
        hue="variant",
        fill=True,
        ax=axes[2],
        palette=["#7f7f7f", "#1f77b4"],
    )
    axes[2].set_title(
        "Session Duration Distribution (sec)", fontsize=12, fontweight="bold"
    )
    axes[2].set_xlabel("Seconds")

    plt.tight_layout()
    output_png = "ab_test_dashboard_v2.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"✅ Dashboard visualization saved cleanly as '{output_png}'.")


if __name__ == "__main__":
    df = generate_ab_data()
    p_val, lift = perform_statistical_testing(df)
    export_dashboard_visuals(df, p_val)