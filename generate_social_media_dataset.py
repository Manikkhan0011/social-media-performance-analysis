from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 1. Reproducibility & Setup
np.random.seed(42)
num_posts = 600

platforms = ["Instagram", "LinkedIn", "Facebook"]
post_types = ["Image", "Video", "Carousel", "Text"]

# 2. Realistic Timeline (Past 6 Months Data)
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
random_dates = [
    start_date + timedelta(days=int(np.random.randint(0, 180)))
    for _ in range(num_posts)
]

# 3. Data Dictionary Construct Karna
data = {
    "Post_ID": [f"POST_{i+1:04d}" for i in range(num_posts)],
    "Date": [d.strftime("%Y-%m-%d") for d in random_dates],
    "Platform": np.random.choice(platforms, size=num_posts, p=[0.4, 0.4, 0.2]),
    "Post_Type": np.random.choice(
        post_types, size=num_posts, p=[0.35, 0.35, 0.2, 0.1]
    ),
    "Likes": np.random.randint(20, 1500, size=num_posts),
    "Comments": np.random.randint(2, 250, size=num_posts),
    "Shares": np.random.randint(0, 180, size=num_posts),
    "Reach": np.random.randint(500, 10000, size=num_posts),
}

df = pd.DataFrame(data)

# 4. Core Metrics Calculation
df["Total_Engagement"] = df["Likes"] + df["Comments"] + df["Shares"]
df["Engagement_Rate (%)"] = np.round(
    (df["Total_Engagement"] / df["Reach"]) * 100, 2
)

# Date wise sort kar lete hain
df = df.sort_values(by="Date").reset_index(drop=True)

# 5. Dataset Save Karna
csv_filename = "social_media_data.csv"
df.to_csv(csv_filename, index=False)

# 6. Professional Console Summary Output
print("=" * 60)
print("🚀 SOCIAL MEDIA PERFORMANCE DATASET GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"📌 Total Records Generated: {len(df)}")
print(f"📁 Saved File Name        : {csv_filename}\n")

print("📊 PLATFORM ENGAGEMENT SUMMARY:")
summary = df.groupby("Platform")[
    ["Reach", "Total_Engagement", "Engagement_Rate (%)"]
].mean()
print(summary.round(2))

print("\n🔍 DATASET PREVIEW (First 5 Rows):")
print(df.head())
print("=" * 60)