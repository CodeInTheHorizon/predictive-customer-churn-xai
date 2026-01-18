# Predictive Customer Churn Model (Telecom)

## Overview
This project implements an end-to-end machine learning solution to predict customer churn in the telecom industry and explain the reasons behind each prediction using Explainable AI (XAI).

The model identifies customers who are likely to churn and highlights the key factors influencing their decision, enabling businesses to take proactive and targeted retention actions. To ensure transparency and trust, SHAP (SHapley Additive exPlanations) is used to interpret both global model behavior and individual customer predictions.

The solution combines data analysis, feature engineering, model training, optimization, and interpretability into a single reusable pipeline. Multiple machine learning models are evaluated, with XGBoost selected as the final model based on AUC-ROC performance. The trained pipeline can be directly applied to new customer data without additional preprocessing.

Key insights from the model show that contract type, customer tenure, and monthly charges are the strongest drivers of churn. Customers on month-to-month contracts and those with high monthly charges are at the highest risk, while long-term contracts, longer tenure, and value-added services significantly reduce churn probability.

This project demonstrates how predictive modeling and explainable AI can be combined to create business-ready machine learning systems suitable for real-world deployment in customer retention and CRM workflows.

---

## Technologies Used
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
