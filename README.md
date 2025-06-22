❤️ Heart Disease Prediction
🩺 Project Overview
Given a set of clinical parameters about a patient — such as age, blood pressure, cholesterol levels, and more — can we predict whether or not they have heart disease?

This project aims to answer that question using various machine learning classification algorithms from the Scikit-learn library. It is inspired by real-world health data and demonstrates the application of ML in healthcare diagnostics.

📊 Dataset
The dataset used is a well-known heart disease dataset available through UCI Machine Learning Repository. It contains features like:

Age

Sex

Resting blood pressure

Serum cholesterol

Fasting blood sugar

Maximum heart rate achieved

Exercise-induced angina

ST depression, etc.

The target variable is target:

1 indicates presence of heart disease

0 indicates no heart disease

⚙️ Models Used
Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Each model is evaluated based on accuracy and performance metrics like confusion matrix, precision, recall, and F1-score.

🧠 Goal
To build a reliable and interpretable model that can assist in the early detection of heart disease and potentially guide medical decisions.

🚀 Features
Data cleaning and preprocessing

Exploratory data analysis (EDA) using matplotlib and seaborn

Model comparison and evaluation

Custom fit_and_score() utility function

Visualizations of prediction results

📦 Requirements
Python 3.x

pandas, numpy

matplotlib, seaborn

scikit-learn
