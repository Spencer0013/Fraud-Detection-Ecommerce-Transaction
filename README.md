## Fraud Detection System

This project is an end-to-end machine learning system for detecting fraudulent transactions in e-commerce data. It includes a modular training pipeline and a Streamlit web application that allows non-technical users to upload transaction files and receive fraud predictions.

## Project Background

E-commerce platforms face significant losses due to fraudulent activity. Detecting these transactions early is challenging because fraud is rare compared to legitimate activity and often blends in with normal behavior.

This project was developed to:

Build a reliable fraud detection model with high recall while maintaining precision.

Handle class imbalance effectively using techniques like SMOTE.

Provide a pipeline that is reproducible and deployment-ready.

Offer a user-friendly interface for fraud detection.

<img width="745" height="507" alt="image" src="https://github.com/user-attachments/assets/ea291ddf-a2f8-40a7-a3b0-5c4b57d12558" />


## Workflow Overview
1. Data Ingestion

Reads raw train/test CSV files.

Converts columns to correct types.

Saves cleaned datasets for use in later stages.

2. Data Transformation

Engineers features relevant to fraud detection, including:

Time-based features (day of week, weekend flag, transaction hour bins).

Customer/account signals (account age bins, new account flag).

Transaction markers (log-transformed amounts, high-value flags).

Address and IP mismatches.

Encodes categorical variables and applies preprocessing.

Splits into training, validation, and test sets.

3. Model Training

Trains multiple models: Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, AdaBoost, and Gradient Boosting.

Uses SMOTE to address class imbalance.

Evaluates each model using precision, recall, F1, and ROC-AUC.

Selects the best baseline model.

4. Model Tuning

Runs hyperparameter tuning with RandomizedSearchCV, using ROC-AUC as the scoring metric.

Supports tuning for Random Forest, AdaBoost, Gradient Boosting, and LightGBM.

Saves the best model and its parameters.

5. Model Evaluation

Evaluates the tuned model on a validation set.

Records accuracy, ROC-AUC, confusion matrix, and per-class metrics.

Saves evaluation results as structured JSON for reproducibility.

## Results

The final tuned model achieved the following results on the validation set:

Best model: LightGBM

Accuracy: ~98.5%

Fraud class (label = 1):

Precision: 0.96

Recall: 0.94

F1-score: 0.95

Non-fraud class (label = 0):

Precision: 0.99

Recall: 0.99

ROC-AUC: 0.99

These results show that the model is effective at identifying fraud while minimizing false alarms.

## Streamlit Application

The included web application allows users to:

Upload a CSV file of transactions.

Automatically preprocess and transform the data.

Apply the trained model to generate fraud predictions.

Review predictions in the browser.

Download results with predictions included.

Tech Stack

Python (pandas, numpy, scikit-learn, imbalanced-learn)

Gradient boosting frameworks (LightGBM, XGBoost, CatBoost)

Streamlit (web interface)

Kubernetes (deployment)

Summary

This project demonstrates a complete machine learning workflow for fraud detection in e-commerce. It combines data processing, feature engineering, model selection, tuning, evaluation, and deployment into a reproducible system. The resulting model achieves high accuracy and recall, making it a strong candidate for real-world fraud detection scenarios.
