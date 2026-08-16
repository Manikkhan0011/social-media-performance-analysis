# 📊 LMS User Churn Prediction Model

A Machine Learning project to predict user churn on Learning Management Systems (LMS) using engagement logs and automated retention strategies.

## 🏗️ Project Architecture

```mermaid
graph TD
    A[LMS Engagement Logs] --> B[Data Preprocessing & Feature Engineering]
    B --> C[Engagement Score & Scaling]
    C --> D[Model Training: Random Forest & Gradient Boosting]
    D --> E[Model Evaluation: ROC-AUC & Confusion Matrix]
    E --> F[High-Risk Churn Prediction]
    F --> G[Personalized Retention Support Strategies]
```

## 🚀 How to Run
```bash
python churn_prediction_model.py
```