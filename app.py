
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("ML Classification Model Comparison")
st.write(
    "An interactive application to evaluate "
    "five machine learning classification models."
)

st.divider()

# -------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Support Vector Machine": "models/svm.pkl"
}

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_models():

    loaded = {}

    for name, path in model_files.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded[name] = joblib.load(path)

    return loaded


try:
    models = load_models()

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# -------------------------------------------------
# FEATURE CONFIGURATION
# -------------------------------------------------

feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean', 'compactness_mean',
    'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        required_columns = feature_columns + ['diagnosis']

        missing_columns = [
            col for col in required_columns
            if col not in test_df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        if test_df.empty:
            st.error("Uploaded dataset is empty.")
            st.stop()

        if test_df[required_columns].isnull().any().any():
            st.error(
                "Dataset contains missing values. "
                "Please upload a valid test dataset."
            )
            st.stop()

        X_test = test_df[feature_columns].copy()

        if not all(
            pd.api.types.is_numeric_dtype(X_test[col])
            for col in feature_columns
        ):
            st.error("All input features must be numeric.")
            st.stop()

        # Encode target
        y_test = test_df['diagnosis'].replace({
            'B': 0,
            'M': 1
        })

        if not set(y_test.unique()).issubset({0, 1}):
            st.error(
                "Target column must contain only B/M or 0/1."
            )
            st.stop()

        y_test = y_test.astype(int)

        if len(y_test.unique()) != 2:
            st.error(
                "Both target classes are required to calculate AUC."
            )
            st.stop()

        st.success("Dataset uploaded successfully!")

        st.write("Dataset shape:", test_df.shape)

        with st.expander("Preview uploaded dataset"):
            st.dataframe(test_df.head())

        st.divider()

        # -------------------------------------------------
        # MODEL SELECTION
        # -------------------------------------------------

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a model",
            list(models.keys())
        )

        model = models[selected_model]

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # -------------------------------------------------
        # EVALUATION METRICS
        # -------------------------------------------------

        st.header("3. Model Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_prob)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.divider()

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(selected_model)

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=['Benign', 'Malignant'],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose()
        )

        # -------------------------------------------------
        # PREDICTION RESULTS
        # -------------------------------------------------

        st.header("6. Prediction Results")

        prediction_results = test_df.copy()

        prediction_results['Actual'] = y_test.map({
            0: 'Benign',
            1: 'Malignant'
        }).values

        prediction_results['Predicted'] = pd.Series(
            y_pred
        ).map({
            0: 'Benign',
            1: 'Malignant'
        }).values

        prediction_results['Prediction Probability'] = y_prob

        st.dataframe(prediction_results)

        st.download_button(
            "Download Prediction Results",
            prediction_results.to_csv(index=False),
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error processing dataset: {e}")

else:

    st.info(
        "Please upload test_data.csv to evaluate the models."
    )

st.divider()

st.caption(
    "ML Assignment 2 | M.Tech AIML | Machine Learning"
)
