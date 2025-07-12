# Fraud Detection on E-commerce Transactions

This repository focuses on detecting fraudulent transactions in e-commerce using advanced machine learning techniques. The solution combines extensive feature engineering, state-of-the-art models, and robust evaluation to help minimize fraudulent activities and protect business revenue.

---

## 💡 Overview

E-commerce fraud detection is critical to maintaining customer trust and preventing financial losses. This project provides an end-to-end pipeline including data ingestion, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

---


---

## 📄 Dataset

The dataset includes:

- Transaction metadata (e.g., amount, time, payment method)
- Customer information (age, location, account age)
- Behavioral and risk indicators (address match, device used, IP address)
- Labels indicating if a transaction is fraudulent

---

## 🗂️ Project Structure

FRAUD DETECTION/
├── artifacts/
├── catboost_info/
├── config/
├── Data/
├── logs/
├── research/
│ ├── 01_data_ingestion.ipynb
│ ├── 02_data_transformation.ipynb
│ ├── 03_model_trainer.ipynb
│ ├── 04_model_tuner.ipynb
│ ├── 05_model_evaluation.ipynb
│ └── trials.ipynb
├── src/
│ ├── fraud_detection/
│ │ ├── components/
│ │ │ ├── data_ingestion.py
│ │ │ ├── data_transformation.py
│ │ │ ├── model_trainer.py
│ │ │ ├── model_tuner.py
│ │ │ └── model_evaluation.py
│ │ ├── config/
│ │ ├── constants/
│ │ └── entity/
│ └── pipeline/
│ ├── stage_01_data_ingestion.py
│ ├── stage_02_data_transformation.py
│ ├── stage_03_model_trainer.py
│ ├── stage_04_model_tuner.py
│ └── stage_05_model_evaluation.py
├── app.py
├── main.py
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── template.py
├── setup.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── README.md

---

## ⚙️ Pipeline Components

### ✅ Data Ingestion

- Reads and cleans raw training and testing data from CSV files
- Converts data types and parses dates

> Implemented in [`data_ingestion.py`](./data_ingestion.py)

---

### 🧬 Data Transformation

- Extensive feature engineering: log-transformed amounts, temporal bins, address mismatches, IP-derived features, customer behavior indicators, etc.
- Preprocessing using scikit-learn pipelines (numerical scaling, categorical encoding)
- Automated split into train, validation, and test sets

> Implemented in [`data_transformation.py`](./data_transformation.py)

---

### 🏋️ Model Training

- Models trained include Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, AdaBoost, and Gradient Boosting
- Handles severe class imbalance using SMOTE
- Evaluates each model on validation data using precision, recall, F1 score, and ROC AUC
- Selects the best-performing model and saves it

> Implemented in [`model_trainer.py`](./model_trainer.py)

---

### 🔧 Hyperparameter Tuning

- Supports Random Forest, AdaBoost, Gradient Boosting, and LightGBM
- Uses randomized search with cross-validation and ROC AUC scoring to find optimal hyperparameters

> Implemented in [`model_tuner.py`](./model_tuner.py)

---

### 📊 Model Evaluation

- Evaluates the best model on validation data
- Outputs detailed metrics including accuracy, confusion matrix, ROC AUC, and per-class precision, recall, and F1 score
- Saves structured evaluation results in JSON format

> Implemented in [`model_evaluation.py`](./model_evaluation.py)

---

## 🏆 Evaluation Results

Results (from `evaluation_results.json`):

- **Validation Accuracy**: 95.5%
- **Validation ROC AUC**: 0.811
- **Confusion Matrix**:

[[279307, 516],
[12602, 2166]]

- **Non-fraud (Class 0)**:
- Precision: 0.96
- Recall: 99.8%
- F1 Score: 0.98
- **Fraud (Class 1)**:
- Precision: 0.81
- Recall: 14.7%
- F1 Score: 0.25

While the recall for fraud cases is low (common in imbalanced fraud settings), high precision suggests flagged cases are highly likely to be true fraud.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Spencer0013/Fraud-Detection-Ecommerce-Transaction.git
cd Fraud-Detection-Ecommerce-Transaction
pip install -r requirements.txt

📈 Future Work
Improve fraud recall via advanced sampling or ensemble techniques

Add real-time detection capabilities (API or streaming)

Deploy best model as a microservice

Implement continuous monitoring for model drift
