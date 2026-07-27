import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Visual style setup
sns.set_theme(style="whitegrid")

# 1. Dataset Load Karna
df = pd.read_csv("social_media_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("🔍 Analyzing Social Media Performance Metrics...\n")

# --- CHART 1: Engagement Rate by Platform ---
plt.figure(figsize=(8, 5))
platform_engagement = (
    df.groupby("Platform")["Engagement_Rate (%)"]
    .mean()
    .reset_index()
    .sort_values(by="Engagement_Rate (%)", ascending=False)
)

# FIXED: Warning removed by adding hue="Platform" and legend=False
ax = sns.barplot(
    data=platform_engagement,
    x="Platform",
    y="Engagement_Rate (%)",
    hue="Platform",
    palette="Blues_d",
    legend=False,
)
plt.title(
    "Average Engagement Rate (%) by Platform", fontsize=14, fontweight="bold"
)
plt.xlabel("Platform", fontsize=12)
plt.ylabel("Engagement Rate (%)", fontsize=12)

# Values labels add karna
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 8),
        textcoords="offset points",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("platform_engagement_rate.png", dpi=300)
plt.close()
print("✅ Saved Chart 1: platform_engagement_rate.png")

# --- CHART 2: Performance by Post Type ---
plt.figure(figsize=(9, 5))
post_type_perf = (
    df.groupby(["Platform", "Post_Type"])["Total_Engagement"]
    .mean()
    .reset_index()
)

sns.barplot(
    data=post_type_perf,
    x="Platform",
    y="Total_Engagement",
    hue="Post_Type",
    palette="viridis",
)
plt.title(
    "Average Total Engagement by Post Type across Platforms",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Platform", fontsize=12)
plt.ylabel("Avg Total Engagement", fontsize=12)
plt.legend(title="Post Type")

plt.tight_layout()
plt.savefig("post_type_performance.png", dpi=300)
plt.close()
print("✅ Saved Chart 2: post_type_performance.png")

# --- CHART 3: Monthly Reach & Engagement Trend ---
df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_trend = (
    df.groupby("Year_Month")[["Reach", "Total_Engagement"]].sum().reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5))

color = "tab:blue"
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Total Reach", color=color, fontsize=12)
ax1.plot(
    monthly_trend["Year_Month"],
    monthly_trend["Reach"],
    color=color,
    marker="o",
    linewidth=2.5,
)
ax1.tick_params(axis="y", labelcolor=color)
plt.xticks(rotation=45)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Total Engagement", color=color, fontsize=12)
ax2.plot(
    monthly_trend["Year_Month"],
    monthly_trend["Total_Engagement"],
    color=color,
    marker="s",
    linewidth=2.5,
    linestyle="--",
)
ax2.tick_params(axis="y", labelcolor=color)

plt.title(
    "Overall Monthly Reach & Engagement Trend", fontsize=14, fontweight="bold"
)
fig.tight_layout()
plt.savefig("monthly_performance_trend.png", dpi=300)
plt.close()
print("✅ Saved Chart 3: monthly_performance_trend.png")

print(
    "\n🚀 ANALYSIS COMPLETE! Check your folder for the generated image files."
)