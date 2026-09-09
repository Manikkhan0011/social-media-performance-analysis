# ==============================================================================
# Task 5: A/B Testing Analytics Engine (Clean Image Export Fix)
# Internee.pk Data Science Internship
# ==============================================================================

import matplotlib
matplotlib.use("Agg")  # Non-interactive clean backend to fix corrupt file header
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def generate_ab_data(n_samples=10000):
    np.random.seed(42)
    n_control = n_samples // 2
    n_treatment = n_samples // 2

    conv_a = np.random.binomial(1, p=0.10, size=n_control)
    bounce_a = np.random.binomial(1, p=0.48, size=n_control)
    duration_a = np.random.exponential(scale=110, size=n_control)

    conv_b = np.random.binomial(1, p=0.128, size=n_treatment)
    bounce_b = np.random.binomial(1, p=0.39, size=n_treatment)
    duration_b = np.random.exponential(scale=140, size=n_treatment)

    df_a = pd.DataFrame(
        {
            "user_id": [f"user_A_{i}" for i in range(n_control)],
            "variant": "Control (A)",
            "converted": conv_a,
            "bounced": bounce_a,
            "session_duration_sec": np.round(duration_a, 2),
        }
    )

    df_b = pd.DataFrame(
        {
            "user_id": [f"user_B_{i}" for i in range(n_treatment)],
            "variant": "Treatment (B)",
            "converted": conv_b,
            "bounced": bounce_b,
            "session_duration_sec": np.round(duration_b, 2),
        }
    )

    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_csv("ab_testing_analytics_data.csv", index=False)
    return df


def perform_statistical_testing(df):
    conv_a = df[df["variant"] == "Control (A)"]["converted"]
    conv_b = df[df["variant"] == "Treatment (B)"]["converted"]

    p1, p2 = conv_a.mean(), conv_b.mean()
    p_pooled = (conv_a.sum() + conv_b.sum()) / (len(conv_a) + len(conv_b))
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / len(conv_a) + 1 / len(conv_b)))
    z_stat = (p2 - p1) / se
    p_val_conv = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return p_val_conv


def export_dashboard_visuals(df, p_val_conv):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    plt.tight_layout()

    # Universally supported JPG export
    output_img = "ab_test_dashboard_final.jpg"
    plt.savefig(output_img, dpi=300, bbox_inches="tight", format="jpeg")
    plt.close("all")
    print(f"✅ Clean visual image exported successfully as '{output_img}'.")


if __name__ == "__main__":
    df = generate_ab_data()
    p_val = perform_statistical_testing(df)
    export_dashboard_visuals(df, p_val)