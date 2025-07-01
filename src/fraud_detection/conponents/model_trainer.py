import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from fraud_detection.entity import DataTransformationConfig
from fraud_detection.conponents.data_transformation import DataTransformation
from fraud_detection.utils.common import save_object
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE



class ModelTrainer:
    def __init__(self, config, data_transformer):
        self.config = config
        self.data_transformer = data_transformer

    def train(self):
        # Get train/val/test splits from data transformer
        (
            X_train,   # after preprocessing and SMOTE will be applied here
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            preprocessor_path
        ) = self.data_transformer.initiate_data_transformation_and_split()

        # Apply SMOTE only on training data
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        models = {

            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                                  scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),  # xgboost way
            "CatBoostClassifier": CatBoostClassifier(verbose=False, random_state=42, auto_class_weights='Balanced'),  # catboost way
            "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=42),  # no class_weight param
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42,  # no class_weight param
                                                               )
            }

        best_model = None
        best_model_name = None
        best_auc = 0
        scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_val)
            else:
                y_proba = y_pred

            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_val, y_proba)
            except Exception:
                auc = 0

            scores[name] = {
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
            }

            print(f"[ModelTrainer] {name} Metrics:")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC AUC:   {auc:.4f}")
            print(f"  Classification Report:\n{classification_report(y_val, y_pred, zero_division=0)}")
            print("-" * 60)

            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_model_name = name

        print(f"[ModelTrainer] Best Model: {best_model_name} | Best ROC AUC: {best_auc:.4f}")

        if self.config.model_save_path:
            save_object(self.config.model_save_path, best_model)
            print(f"[ModelTrainer] Best model saved to: {self.config.model_save_path}")

        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_roc_auc": best_auc,
            "all_scores": scores,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "preprocessor_path": preprocessor_path
        }
