# Machine Learning Classification Model Comparison

## a. Problem Statement

The objective of this assignment is to implement and compare five different machine learning classification algorithms using a common classification dataset.

The Wisconsin Diagnostic Breast Cancer dataset is used to predict whether a breast tumor is Benign or Malignant based on various numerical features extracted from digitized images of breast mass samples.

The five classification algorithms are evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to demonstrate the trained models and evaluate their performance using uploaded test data.

## b. Dataset Description

**Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Dataset Characteristics:**

| Property | Description |
|---|---|
| Total Instances | 569 |
| Total Input Features | 30 |
| Target Variable | Diagnosis |
| Classification Type | Binary Classification |
| Target Classes | Benign (B), Malignant (M) |
| Missing Values | 0 |

The dataset contains 30 numerical features computed from digitized images of breast mass samples. These features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

**Target Variable Encoding:**

- Benign (B) = 0
- Malignant (M) = 1

The ID column is removed because it does not contribute to classification.

**Data Preprocessing:**

- Removed the unnecessary ID column.
- Encoded the diagnosis target variable.
- Separated independent and dependent variables.
- Split the dataset into 80% training data and 20% testing data.
- Applied StandardScaler for models requiring feature scaling.
- Used stratified sampling to maintain class distribution.

## c. GitHub Repository Link

**GitHub Repository:**

https://github.com/Karthikeyan-BITS-2026ak05465/ml-classification-model-comparison

The repository contains the complete source code, Jupyter Notebook, trained models, test dataset, requirements file, README, and Streamlit application.

## d. Models Used

The following five machine learning classification algorithms were implemented using Python and Scikit-learn:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Performance Comparison

All five models were evaluated using the same testing dataset. The following table presents the calculated evaluation metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | 0.9286 | **0.9630** | **0.9442** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 96.49% and the highest AUC score of 0.9960 among all five models. It also achieved a precision of 97.50% and an F1 score of 0.9512, demonstrating strong and balanced classification performance. |
| Decision Tree | Achieved an accuracy of 92.98%, with precision and recall of 90.48%. It recorded the lowest AUC score of 0.9246 and MCC score of 0.8492 among the five models. Its performance was comparatively lower on this dataset. |
| KNN | Achieved an accuracy of 95.61% and an AUC score of 0.9823. Its precision was 97.44%, while recall was 90.48%. It demonstrated good classification performance, although its recall was lower than Logistic Regression and SVM. |
| Naive Bayes | Achieved an accuracy of 93.86% and an AUC score of 0.9934. It recorded 100% precision, indicating that all predicted positive cases were correct in the test set. However, its recall of 83.33% was the lowest among the five models. |
| Support Vector Machine | Achieved the highest accuracy of 97.37%, an F1 score of 0.9630, and the highest MCC score of 0.9442. It also achieved 100% precision and 92.86% recall, demonstrating the strongest overall classification performance among the five models. |

### Overall, Winner for your dataset?

**Support Vector Machine (SVM)**

Support Vector Machine is the overall best-performing model for the Wisconsin Diagnostic Breast Cancer dataset.

It achieved the highest accuracy of 97.37%, an F1 score of 0.9630, and an MCC score of 0.9442 among all five classification models. It also achieved 100% precision and 92.86% recall.

Although Logistic Regression achieved the highest AUC score of 0.9960, SVM demonstrated a better overall balance across the evaluation metrics.

Therefore, Support Vector Machine is selected as the overall winner for this dataset.

## Streamlit Web Application

**Live Application Link:**

https://ml-classification-model-comparison-zj7xpp9eygtigxyn7zvizz.streamlit.app/

The interactive Streamlit application provides the following functionalities:

1. Upload a test dataset in CSV format.
2. Select one of the five trained classification models using a dropdown.
3. Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. Display the confusion matrix.
5. Display the classification report.
6. View actual and predicted classifications.
7. Download prediction results as a CSV file.

The application loads previously trained models saved using Joblib and evaluates the uploaded test dataset without retraining the models.
