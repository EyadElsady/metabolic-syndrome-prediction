# Metabolic Syndrome Prediction & Classification

This repository contains a data science and machine learning workflow designed to predict whether an individual has **Metabolic Syndrome** based on clinical, demographic, and laboratory data. 

The project involves comprehensive data preprocessing, feature engineering, and the training of multiple classification models to optimize performance (ROC-AUC score).

## 📊 Project Overview
Metabolic syndrome is a cluster of conditions—such as increased blood pressure, high blood sugar, excess body fat around the waist, and abnormal cholesterol levels—that occur together, increasing the risk of heart disease, stroke, and type 2 diabetes. 

This project cleanses patient records, handles missing data, normalizes features, and builds predictive models to estimate the risk probability of a patient having the syndrome.

## 🗂️ Repository Structure
* `Metabolic_Syndrome_Project.ipynb`: Jupyter Notebook containing the full pipeline: Data exploration (EDA), handling missing values, encoding categorical variables, pipeline scaling, model training, evaluation, and final inference.
* `train.csv`: Historical dataset containing clinical metrics and the target flag `MetabolicSyndrome`.
* `test.csv`: Independent evaluation dataset used for verifying final model performance.
* `Project2_Submission_Final.csv`: The final generated output file containing the predicted probabilities (`predict_proba`) mapped to patient IDs.

## 🧬 Features & Clinical Attributes
The dataset incorporates the following key patient dimensions:
* **Demographics:** Age, Sex, Marital Status, Income, Race
* **Physical Exams:** Waist Circumference (`WaistCirc`), Body Mass Index (`BMI`)
* **Laboratory Metrics:** Albuminuria, Urinary Albumin-to-Creatinine Ratio (`UrAlbCr`), Uric Acid, Blood Glucose, HDL Cholesterol, Triglycerides
* **Target Label:** `MetabolicSyndrome` (0 = No Syndrome, 1 = Has Syndrome)

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
    * *Preprocessing:* `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`
    * *Models Evaluated:* K-Nearest Neighbors (KNN Classifier), Gaussian Naive Bayes (GaussianNB)
* **Visualization:** `matplotlib` (ROC Curve analysis)

## 🚀 Workflow Details
1.  **Data Preprocessing:** Implements a robust `ColumnTransformer` pipeline. Numeric values handle missing fields using median imputation followed by standard scaling. Categorical attributes are handled via most-frequent imputation and One-Hot Encoding.
2.  **Model Exploration:** Evaluation and comparison between **K-Nearest Neighbors (KNN)** and **Naive Bayes**.
3.  **Optimization:** Utilizes class probability estimations (`predict_proba`) to establish fine-grained thresholds, maximizing the Area Under the ROC Curve (ROC-AUC).
4.  **Submission:** Generates a compliant prediction mapping file (`Project2_Submission_Final.csv`) for submission evaluations.

## 📈 Key Visualizations Included
* **ROC Curve Comparison:** Graphical plot highlighting the true positive vs. false positive rates between the trained KNN and Naive Bayes models to ensure the selection of the strongest classifier.

---
*Developed as part of a Data Science Portfolio Project.*
