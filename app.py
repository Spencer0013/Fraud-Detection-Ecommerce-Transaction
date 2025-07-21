import streamlit as st
import pandas as pd
import joblib
import os
import tempfile

from fraud_detection.conponents.data_ingestion import DataIngestion
from fraud_detection.conponents.data_transformation import DataTransformation
from fraud_detection.entity import DataIngestionConfig, DataTransformationConfig
from fraud_detection.utils.common import read_yaml
from fraud_detection.constants import CONFIG_FILE_PATH

# Load config & init components 
config = read_yaml(CONFIG_FILE_PATH)
ingest_cfg = DataIngestionConfig(**config.data_ingestion)
trans_cfg  = DataTransformationConfig(**config.data_transformation)

data_ingestion     = DataIngestion(ingest_cfg)
data_transformation = DataTransformation(trans_cfg)

# Streamlit page setup 
st.set_page_config(page_title="Fraud Detection")
st.title("E‑commerce Fraud Detection")
st.markdown("Upload a CSV of transactions; the model will flag fraud (0 = legit, 1 = fraud).")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    try:
        # Read raw upload for display
        df_display = pd.read_csv(uploaded_file, low_memory=False, parse_dates=['Transaction Date'])
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_display.head(5))

        # Convert types in memory
        df_clean = data_ingestion.convert_data_types(df_display.copy())

        # Write cleaned DF to a temporary CSV so your process_file (path‑based) works
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, "streamlit_temp.csv")
        df_clean.to_csv(tmp_path, index=False)

        #  Feature‑engineer & select columns via your unchanged process_file
        X, _ = data_transformation.process_file(tmp_path)

        # Load preprocessor & transform
        preprocessor = joblib.load(trans_cfg.preprocessor)
        X_processed = preprocessor.transform(X)

        # Load model & predict
        model = joblib.load(config.model_tuner.tuner_save_path)
        preds = model.predict(X_processed)

        # Attach predictions to original DF
        df_display['Prediction'] = preds

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(df_display[['Transaction Date',
                                 'Transaction Amount',
                                 'Payment Method',
                                 'Product Category',
                                 'Prediction']])

        # Summary
        total = len(df_display)
        frauds = int(df_display['Prediction'].sum())
        st.markdown(f"**Summary:** {frauds} frauds out of {total} transactions.")

        # Download
        csv_out = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Predictions",
                           data=csv_out,
                           file_name="fraud_predictions.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"Processing error: {e}")

