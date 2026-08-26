# ==============================================================================
# Task 4: Competitor Benchmarking & Pricing Dashboard
# Internee.pk Data Science Internship
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual aesthetic theme
sns.set_theme(style="darkgrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ------------------------------------------------------------------------------
# STEP 1: Web Scraping Simulation / Competitor Data Aggregation
# ------------------------------------------------------------------------------
print("=== Step 1: Aggregating Competitor Benchmarking Data ===")

competitors_data = {
    "Competitor": [
        "EduTech Pro",
        "SkillNation",
        "LearnHub",
        "AcademyX",
        "Internee.pk (Us)",
    ],
    "Monthly_Subscription_USD": [29.99, 49.99, 19.99, 39.99, 14.99],
    "Courses_Offered": [150, 320, 85, 210, 180],
    "Avg_User_Rating": [4.3, 4.7, 4.0, 4.5, 4.8],
    "Active_Users_Thousands": [45, 120, 25, 75, 90],
    "Certificate_Included": ["Yes", "Yes", "No", "Yes", "Yes"],
    "Mentorship_Support": ["No", "Yes", "No", "No", "Yes"],
}

df_comp = pd.DataFrame(competitors_data)

# Engineered Metric: Price-to-Value Index (Lower price + higher rating = better value score)
df_comp["Value_Score"] = np.round(
    (df_comp["Avg_User_Rating"] * df_comp["Courses_Offered"])
    / df_comp["Monthly_Subscription_USD"],
    2,
)

print("\nCompetitor Data Table:\n")
print(df_comp.to_string(index=False))

# Save aggregated dataset
df_comp.to_csv("competitor_benchmarking_data.csv", index=False)
print(
    "\nData saved successfully to 'competitor_benchmarking_data.csv'."
)

# ------------------------------------------------------------------------------
# STEP 2: Multi-Panel Visual Benchmarking Dashboard
# ------------------------------------------------------------------------------
print("\n=== Step 2: Generating Benchmarking Dashboard Visuals ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Competitor Benchmarking & Market Positioning Dashboard",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Palette highlight for our brand
colors = [
    "#808080" if comp != "Internee.pk (Us)" else "#2ca02c"
    for comp in df_comp["Competitor"]
]

# Plot 1: Pricing Comparison
sns.barplot(
    data=df_comp,
    x="Competitor",
    y="Monthly_Subscription_USD",
    hue="Competitor",
    legend=False,
    ax=axes[0, 0],
    palette=colors,
)
axes[0, 0].set_title(
    "1. Monthly Subscription Price ($ USD)", fontsize=12, fontweight="bold"
)
axes[0, 0].set_ylabel("Price ($)")
for p in axes[0, 0].patches:
    axes[0, 0].annotate(
        f"${p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 8),
        textcoords="offset points",
        fontweight="bold",
    )

# Plot 2: Courses Offered vs Active Users
sns.scatterplot(
    data=df_comp,
    x="Courses_Offered",
    y="Active_Users_Thousands",
    size="Avg_User_Rating",
    hue="Competitor",
    sizes=(100, 400),
    ax=axes[0, 1],
    palette="viridis",
)
axes[0, 1].set_title(
    "2. Market Reach: Courses Offered vs Active Users",
    fontsize=12,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Number of Courses Offered")
axes[0, 1].set_ylabel("Active Users (in Thousands)")
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot 3: Calculated Value-for-Money Score
sns.barplot(
    data=df_comp.sort_values(by="Value_Score", ascending=False),
    x="Value_Score",
    y="Competitor",
    hue="Competitor",
    legend=False,
    ax=axes[1, 0],
    palette="Blues_r",
)
axes[1, 0].set_title(
    "3. Value-for-Money Score (Higher is Better)",
    fontsize=12,
    fontweight="bold",
)
axes[1, 0].set_xlabel("Value Index Score")

# Plot 4: Pricing vs Rating Positioning Matrix
sns.scatterplot(
    data=df_comp,
    x="Monthly_Subscription_USD",
    y="Avg_User_Rating",
    hue="Competitor",
    s=250,
    ax=axes[1, 1],
    palette="deep",
)
axes[1, 1].set_title(
    "4. Market Positioning: Price vs User Rating",
    fontsize=12,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Monthly Price ($)")
axes[1, 1].set_ylabel("Average User Rating (out of 5)")
axes[1, 1].axvline(
    df_comp["Monthly_Subscription_USD"].mean(),
    color="red",
    linestyle="--",
    alpha=0.6,
    label="Avg Market Price",
)
axes[1, 1].axhline(
    df_comp["Avg_User_Rating"].mean(),
    color="blue",
    linestyle="--",
    alpha=0.6,
    label="Avg Rating",
)
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

# Save image automatically
plt.savefig(
    "competitor_benchmarking_dashboard.png", dpi=300, bbox_inches="tight"
)
print("Dashboard plot saved as 'competitor_benchmarking_dashboard.png'.")
plt.show()

# ------------------------------------------------------------------------------
# STEP 3: Data-Driven Pricing Strategy Recommendations
# ------------------------------------------------------------------------------
print("\n=== Strategic Recommendations for Market Dominance ===")
print("1. Competitive Advantage: Our platform offers the highest Value-Score due to affordable pricing ($14.99) and top user rating (4.8).")
print("2. Mentorship Upsell Opportunity: Only 2 out of 5 platforms provide 1-on-1 mentorship. Highlighting this feature can increase retention.")
print("3. Tiered Pricing Suggestion: Introduce a $24.99 Pro Tier with premium certificates to capture high-value users while remaining below competitor market average ($32.99).")