# ==============================================================================
# Task 3: Customer Churn Prediction Model for LMS Platform
# Internee.pk Data Science Internship
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set aesthetic visual theme for plots
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ------------------------------------------------------------------------------
# STEP 1: Synthetic Dataset Generation (Simulating LMS Engagement Logs)
# ------------------------------------------------------------------------------
print("=== Step 1: Generating LMS User Engagement Logs ===")
np.random.seed(42)
n_users = 1000

user_ids = [f"USER_{1000 + i}" for i in range(n_users)]
login_frequency_per_month = np.random.poisson(lam=12, size=n_users)
course_completion_rate = np.random.uniform(low=0.0, high=1.0, size=n_users)
avg_session_duration_min = np.random.normal(loc=35, scale=12, size=n_users).clip(
    5, 120
)
assignments_submitted = np.random.randint(0, 15, size=n_users)
quiz_avg_score = np.random.uniform(30, 100, size=n_users)
forum_posts_count = np.random.poisson(lam=2, size=n_users)
days_since_last_login = np.random.exponential(scale=15, size=n_users).astype(
    int
)

# Define Churn Probability Logic (Engagement factors)
churn_logit = (
    0.08 * days_since_last_login
    - 0.15 * login_frequency_per_month
    - 2.5 * course_completion_rate
    - 0.1 * assignments_submitted
    + 1.0
)
churn_probability = 1 / (1 + np.exp(-churn_logit))
churned = (churn_probability > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame(
    {
        "User_ID": user_ids,
        "Login_Frequency_Per_Month": login_frequency_per_month,
        "Course_Completion_Rate": np.round(course_completion_rate, 2),
        "Avg_Session_Duration_Min": np.round(avg_session_duration_min, 1),
        "Assignments_Submitted": assignments_submitted,
        "Quiz_Avg_Score": np.round(quiz_avg_score, 1),
        "Forum_Posts_Count": forum_posts_count,
        "Days_Since_Last_Login": days_since_last_login,
        "Churned": churned,
    }
)

print(
    f"Dataset created successfully with {df.shape[0]} rows and {df.shape[1]} columns."
)
print(f"Churn Rate in Dataset: {df['Churned'].mean() * 100:.1f}%\n")
print(df.head())

# ------------------------------------------------------------------------------
# STEP 2: Feature Engineering & Preprocessing
# ------------------------------------------------------------------------------
print("\n=== Step 2: Feature Engineering & Data Splitting ===")

# Create engineered features
df["Engagement_Score"] = (
    df["Login_Frequency_Per_Month"] * 0.4
    + df["Assignments_Submitted"] * 0.3
    + df["Forum_Posts_Count"] * 0.3
)

features = [
    "Login_Frequency_Per_Month",
    "Course_Completion_Rate",
    "Avg_Session_Duration_Min",
    "Assignments_Submitted",
    "Quiz_Avg_Score",
    "Forum_Posts_Count",
    "Days_Since_Last_Login",
    "Engagement_Score",
]

X = df[features]
y = df["Churned"]

# Train-Test Split (80/20 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# STEP 3: Model Training (Random Forest vs Gradient Boosting)
# ------------------------------------------------------------------------------
print("\n=== Step 3: Training Random Forest & Gradient Boosting Models ===")

# 1. Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 2. Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Model Evaluation Results ---")
print(f"Random Forest Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Random Forest ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
print("---------------------------------")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Gradient Boosting ROC-AUC:  {roc_auc_score(y_test, y_prob_gb):.4f}\n")

print(
    "Gradient Boosting Classification Report:\n",
    classification_report(y_test, y_pred_gb),
)

# ------------------------------------------------------------------------------
# STEP 4: Visualizations (ROC Curve, Feature Importance, Confusion Matrix)
# ------------------------------------------------------------------------------
plt.figure(figsize=(16, 5))

# Plot 1: ROC Curve Comparison
plt.subplot(1, 3, 1)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})",
    color="#2b5c8f",
    lw=2,
)
plt.plot(
    fpr_gb,
    tpr_gb,
    label=f"Gradient Boosting (AUC = {roc_auc_score(y_test, y_prob_gb):.2f})",
    color="#d95f02",
    lw=2,
)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.title("ROC Curve Comparison", fontsize=12, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

# Plot 2: Feature Importance (Gradient Boosting)
plt.subplot(1, 3, 2)
importances = gb_model.feature_importances_
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color="#2b5c8f", align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Key Feature Importances (GBM)", fontsize=12, fontweight="bold")
plt.xlabel("Relative Importance")

# Plot 3: Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (Gradient Boosting)", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# STEP 5: Personalized Support Strategy for At-Risk Users
# ------------------------------------------------------------------------------
print("\n=== Step 5: Actionable Personalized Support Strategy ===")
df["Predicted_Churn_Prob"] = gb_model.predict_proba(X)[:, 1]
high_risk_users = df[df["Predicted_Churn_Prob"] > 0.70].sort_values(
    by="Predicted_Churn_Prob", ascending=False
)

print(
    f"Identified {len(high_risk_users)} high-risk users requiring immediate intervention.\n"
)
print("Sample High-Risk Users for Intervention:")
print(
    high_risk_users[
        [
            "User_ID",
            "Days_Since_Last_Login",
            "Course_Completion_Rate",
            "Predicted_Churn_Prob",
        ]
    ].head(5)
)

print("\n--- Personalized Retention Interventions ---")
print(
    "1. Automated Re-engagement Emails: Triggered when 'Days_Since_Last_Login' > 10 days."
)
print(
    "2. Academic Coaching & Mentorship: Target users with 'Course_Completion_Rate' < 30%."
)
print(
    "3. Gamification & Push Notifications: Offer milestone badges for active quiz attempts."
)