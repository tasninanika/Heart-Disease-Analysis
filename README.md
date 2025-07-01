<h1 align="center">Heart Disease Analysis</h1> 
<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Jupyter-FA0F00?style=flat-square&logo=jupyter&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/Matplotlib-007ACC?style=flat-square&logo=plotly&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> </p>

## 📜 Project Overview

The Heart Disease Analysis project is a comprehensive machine learning study aimed at predicting the presence of heart disease using the Heart Disease UCI Dataset. Implemented in a Jupyter Notebook (heart-disease-analysis.ipynb), this project covers data preprocessing, exploratory data analysis (EDA), feature selection, model training, evaluation, and advanced interpretability techniques. It also includes unsupervised learning with Self-Organizing Maps (SOM) for clustering and visualization.
The project employs multiple classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors) and uses techniques like SMOTE for handling class imbalance. Additionally, it integrates interpretability tools (SHAP, LIME) to explain model predictions and geographical visualization with Folium for contextual analysis.

## 🚀 Features

### Data Preprocessing:

* Loads the dataset from heart.csv.
* Handles categorical variables and missing values.
* Applies SMOTE to address class imbalance.

### Exploratory Data Analysis (EDA):

* Summarizes dataset characteristics (e.g., data types, missing values).
* Visualizes feature distributions and correlations.

### Feature Selection:

* Uses mutual information to identify important features.

### Model Training:

* Implements Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN).
* Uses stratified k-fold cross-validation for robust evaluation.

### Model Evaluation:

* Evaluates models using accuracy, confusion matrix, and classification report.

### Model Interpretability:

* Applies SHAP and LIME for explaining model predictions.

### Clustering:

* Uses MiniSom for Self-Organizing Maps to cluster data and visualize patterns.

### Geographical Visualization:

* Integrates Folium and Geopy for mapping hypothetical patient locations.

## 📊 Dataset

The dataset used is the Heart Disease UCI Dataset, containing medical records for heart disease prediction. Key details:

* **Source**: Loaded from heart.csv.

* **Size**: 918 samples, 12 columns.

* **Features**:

  * Age: Patient's age (integer).
  * Sex: Gender (M/F, object).
  * ChestPainType: Type of chest pain (object).
  * RestingBP: Resting blood pressure (integer).
  * Cholesterol: Cholesterol level (integer).
  * FastingBS: Fasting blood sugar (0/1, integer).
  * RestingECG: Resting ECG results (object).
  * MaxHR: Maximum heart rate (integer).
  * ExerciseAngina: Exercise-induced angina (Y/N, object).
  * Oldpeak: ST depression (float).
  * ST\_Slope: Slope of the peak exercise ST segment (object).
  * HeartDisease: Target variable (0 = No disease, 1 = Disease, integer).

* **No Missing Values**: The dataset is clean, with no null values.

### Example data snapshot:

```
Age  Sex  ChestPainType  RestingBP  Cholesterol  FastingBS  RestingECG  MaxHR  ExerciseAngina  Oldpeak  ST_Slope  HeartDisease
40   M    ATA            140         289          0          Normal       172    N               0.0      Up         0
49   F    NAP            160         180          0          Normal       156    N               1.0      Flat       1
...  ...  ...            ...         ...          ...        ...          ...    ...             ...      ...        ...
```

## 📉 Results

The notebook evaluates three classifiers:

* **Logistic Regression**: Linear model for binary classification.
* **Decision Tree Classifier**: Captures non-linear relationships.
* **K-Nearest Neighbors (KNN)**: Distance-based classifier.

              precision    recall  f1-score   support
           0       0.85      0.88      0.86        40
           1       0.90      0.87      0.88        46
      accuracy                               0.87        86
      macro avg          0.87      0.87      0.87        86
      weighted avg       0.88      0.87      0.87        86

### Clustering

A 5x5 Self-Organizing Map (SOM) is used to cluster the data, visualized as a grid showing cluster assignments for samples.

### Interpretability

* **SHAP**: Provides feature importance and contribution to predictions.
* **LIME**: Explains individual predictions with local feature importance.

### Visualizations

* Correlation heatmaps and feature distribution plots for EDA.
* SOM grid for clustering visualization.
* Geographical maps using Folium for hypothetical patient location analysis.


## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.

## 🙌 Acknowledgments

* UCI Machine Learning Repository for providing the Heart Disease UCI Dataset.
* Scikit-learn, TensorFlow, SHAP, LIME, and MiniSom: For machine learning and interpretability tools.
* Pandas, Matplotlib, Seaborn: For data manipulation and visualization.
* Folium and Geopy: For geographical visualizations.
* Jupyter: For an interactive development environment.

