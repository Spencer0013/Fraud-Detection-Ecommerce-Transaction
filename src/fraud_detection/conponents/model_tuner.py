from fraud_detection.utils.common import read_yaml
from fraud_detection.constants import *
from fraud_detection.utils.common import create_directories, save_object
from fraud_detection.entity import DataTransformationConfig, ModelTunerConfig
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier



class ModelTuner:
    def __init__(self, config, data_transformer):
        self.config = config
        self.data_transformer = data_transformer

    def _roc_auc(self, y_true, y_pred_proba):
        """Compute ROC AUC score (probabilities required)."""
        return roc_auc_score(y_true, y_pred_proba[:, 1])

    def tune(self):
        # Unpack with val set included
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor_path = self.data_transformer.initiate_data_transformation_and_split()

        # Define models and param grids from config
        model_name = self.config.model_name
        param_dist = self.config.param_dist.get(model_name, None)

        if param_dist is None:
            raise ValueError(f"[ModelTuner] No param_dist found for {model_name} in config.")

        # Initialize model based on model_name
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == "AdaBoost":
            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(random_state=42)
        elif model_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42, n_iter_no_change=5, validation_fraction=0.1)
        elif model_name == "LightGBM":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"[ModelTuner] Unsupported model: {model_name}")

        scoring = make_scorer(roc_auc_score, needs_proba=True)
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring='roc_auc',
            n_iter=10,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )

        print(f"[ModelTuner] Starting hyperparameter tuning for {model_name}...")

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        print(f"[ModelTuner] Best parameters for {model_name}: {best_params}")

        if self.config.tuner_save_path:
            save_object(self.config.tuner_save_path, best_model)
            print(f"[ModelTuner] Best tuned model saved to: {self.config.tuner_save_path}")

        return best_model, best_params
