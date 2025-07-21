import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import logging
from fraud_detection.utils.common import save_object 
from fraud_detection.entity import DataTransformationConfig


import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Placeholder utility function
def load_and_clean_data(file_path):
    # Implement actual data loading and cleaning
    return pd.read_csv(file_path, low_memory=False)

def save_object(file_path, obj):
    pass



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
      
      """
      Constructs a preprocessing pipeline for numerical and categorical features.

      Parameters:
      - df (pd.DataFrame): Input DataFrame (excluding target column 'Is Fraudulent').

      Returns:
      - ColumnTransformer: A scikit-learn transformer for preprocessing.
      """
      # Drop the target column 
      df = df.drop(columns=["Is Fraudulent"], errors="ignore")

      # Identify numerical and categorical columns
      numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
      categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
  
       # Pipelines with imputation
      numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
        ])

      categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
         ])

    # ColumnTransformer to apply pipelines
      preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
        ])

      return preprocessor

   

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds engineered features to the dataframe."""

         # Convert to datetime, coercing errors to NaT
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    
        print("Transaction Date column after conversion:")
        print(df['Transaction Date'].head())
        print("Data type:", df['Transaction Date'].dtype)
    
        # Transaction Amount Features
        df['Log Transaction Amount'] = np.log1p(df['Transaction Amount'])
        df['Amount Bin'] = pd.qcut(df['Transaction Amount'], q=4, labels=False)

        # Time-based Features
        df['Day of Week'] = df['Transaction Date'].dt.dayofweek
        df['Is Weekend'] = df['Day of Week'].isin([5, 6]).astype(int)
        df['Hour Bin'] = pd.cut(df['Transaction Hour'], bins=[0, 6, 12, 18, 24], labels=False)

        # Address Discrepancy
        df['Address Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)

        # Customer Behavior Features
        df['Amount per Item'] = df['Transaction Amount'] / df['Quantity']
        df['Age Amount Interaction'] = df['Customer Age'] * df['Transaction Amount']
        df['Account Age Bin'] = pd.qcut(df['Account Age Days'], q=4, labels=False)

        # Categorical Encoding
        le = LabelEncoder()
        categorical_cols = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
        for col in categorical_cols:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

        # IP Address Features
        df['IP First Octet'] = df['IP Address'].apply(lambda x: int(x.split('.')[0]) if isinstance(x, str) else 0)

        # Fraud Risk Indicators
        df['High Value Transaction'] = (df['Transaction Amount'] > df['Transaction Amount'].quantile(0.95)).astype(int)
        df['New Account'] = (df['Account Age Days'] < 30).astype(int)

        return df

    def process_file(self, file_path: str):
        """Loads, cleans, engineers features, and returns feature/target split."""
        df = load_and_clean_data(file_path)
        df = self.engineer_features(df)

        feature_columns = [
            'Log Transaction Amount', 'Amount Bin', 'Day of Week', 'Is Weekend', 'Hour Bin',
            'Address Mismatch', 'Amount per Item', 'Age Amount Interaction', 'Account Age Bin',
            'Payment Method_encoded', 'Product Category_encoded', 'Device Used_encoded',
            'Customer Location_encoded', 'IP First Octet', 'High Value Transaction', 'New Account'
        ]
        return df[feature_columns], df['Is Fraudulent']

    def initiate_data_transformation_and_split(self):
        """Main function to load data, preprocess, and return train/test split."""
        # Load, clean, and engineer features
        X_train, y_train = self.process_file(self.config.train_path)
        X_test, y_test = self.process_file(self.config.test_path)

        # Build and apply preprocessing pipeline
        logging.info("Building preprocessing pipeline.")
        preprocessor = self.build_preprocessor(X_train)

        logging.info("Applying preprocessing pipeline.")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Save preprocessor
        save_object(file_path=self.config.preprocessor, obj=preprocessor)

         # Create validation set from training data
        X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        return X_train_final, X_val, X_test_processed, y_train_final, y_val, y_test, self.config.preprocessor
