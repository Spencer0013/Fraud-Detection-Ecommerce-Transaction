from pathlib import Path
from fraud_detection.utils.common import save_json  
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from fraud_detection.conponents.data_transformation import DataTransformation
import joblib  # for loading the model
import os
import json
from fraud_detection.utils.common import read_yaml
from fraud_detection.constants import *
from fraud_detection.utils.common import create_directories, save_object
from fraud_detection.entity import DataTransformationConfig, ModelEvaluationConfig


class ModelEvaluator:
    def __init__(self, config, data_transformer, positive_label=1):
        """
        positive_label: label representing the 'fraud' class, usually 1
        """
        self.config = config
        self.data_transformer = data_transformer
        self.save_path = config.save_path
        self.positive_label = positive_label
        self.model = self._load_model(config.best_model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return joblib.load(model_path)

    def evaluate(self):
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            preprocessor_path
        ) = self.data_transformer.initiate_data_transformation_and_split()

        y_val_pred = self.model.predict(X_val)
        y_val_prob = None
        if hasattr(self.model, "predict_proba"):
            y_val_prob = self.model.predict_proba(X_val)[:, 1]

        # Overall accuracy
        accuracy = accuracy_score(y_val, y_val_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_val, y_val_pred).tolist()

        # Classification report as a dictionary
        report_dict = classification_report(y_val, y_val_pred, output_dict=True)
        fraud_metrics = report_dict[str(self.positive_label)]
        nonfraud_metrics = report_dict[str(1 - self.positive_label)]

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else None
        except ValueError:
            roc_auc = None

        print("=== Classification Report (All Classes) ===")
        print(classification_report(y_val, y_val_pred))

        print("=== Confusion Matrix ===")
        print(conf_matrix)

        print("=== Fraud Class (Label = 1) Metrics ===")
        print(f"Precision: {fraud_metrics['precision']:.4f}")
        print(f"Recall:    {fraud_metrics['recall']:.4f}")
        print(f"F1 Score:  {fraud_metrics['f1-score']:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC:   {roc_auc:.4f}")
        else:
            print("ROC AUC:   N/A")

        # Save structured results
        results = {
            "validation_accuracy": accuracy,
            "validation_confusion_matrix": conf_matrix,
            "validation_roc_auc": roc_auc,
            "class_0_metrics": {
                "precision": nonfraud_metrics['precision'],
                "recall": nonfraud_metrics['recall'],
                "f1_score": nonfraud_metrics['f1-score'],
                "support": nonfraud_metrics['support']
            },
            "class_1_metrics": {
                "precision": fraud_metrics['precision'],
                "recall": fraud_metrics['recall'],
                "f1_score": fraud_metrics['f1-score'],
                "support": fraud_metrics['support']
            }
        }

        if self.save_path:
            save_json(Path(self.save_path), results)
            print(f"[ModelEvaluator] Results saved to {self.save_path}")

        return results