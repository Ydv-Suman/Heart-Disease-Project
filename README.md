❤️ Heart Disease Prediction
Can we predict whether someone has heart disease based on their medical information?

This machine learning project aims to classify whether a patient is likely to have heart disease using clinical features such as age, blood pressure, cholesterol levels, and more.

🔍 Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes.
Given a set of clinical parameters, can we build a machine learning model that accurately predicts the presence of heart disease?

📁 Project Structure
Exploratory Data Analysis (EDA)
Understand data distribution, identify patterns, and detect outliers or missing values.

Data Preprocessing and Cleaning
Handle missing values, encode categorical variables, scale features.

Machine Learning Model Training
Train multiple models and tune hyperparameters.

Model Evaluation and Comparison
Use metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Visualization of Results
Visual tools like confusion matrices, ROC curves, and feature importance plots.

🧪 Models Used
✅ Logistic Regression (Best performance)

🔍 K-Nearest Neighbors (KNN)

🌲 Random Forest Classifier

All models are compared using standard evaluation metrics.
Logistic Regression yielded the best overall performance in terms of accuracy and interpretability.

📊 Dataset
The dataset includes the following clinical features:

Feature	Description
age	Age of the patient
sex	Gender (1 = male, 0 = female)
cp	Chest pain type
trestbps	Resting blood pressure (mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar (> 120 mg/dl)
restecg	Resting electrocardiographic results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina
oldpeak	ST depression induced by exercise
slope, ca, thal	Additional medical measurements
target	Presence of heart disease (1 = yes, 0 = no)

🛠️ Tools & Libraries
Python 3.x

numpy, pandas — Data manipulation

matplotlib, seaborn — Visualization

scikit-learn — Machine learning models and evaluation

📌 Goal
To build a simple yet effective machine learning model for early detection of heart disease using basic patient data.
This tool can potentially assist medical professionals by providing quick, data-driven insights.

✅ Key Findings
Logistic Regression provided the highest accuracy and interpretability.

Proper preprocessing (scaling, encoding) significantly impacted model performance.

Visualizations (ROC Curve, Confusion Matrix) aided in understanding model behavior.

