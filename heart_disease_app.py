import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Changed from joblib to pickle

# Title and description
st.title("Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on input features.")

# Load your trained models using pickle
# Replace these lines with the correct paths to your saved models
with open('C:\\Users\\Quest\\OneDrive\\Desktop\\bmsce\\heart disease prediction\\logistic_regression_model.pkl', 'rb') as file:
    log_reg_model = pickle.load(file)

with open('C:\\Users\\Quest\\OneDrive\\Desktop\\bmsce\\heart disease prediction\\svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('C:\\Users\\Quest\\OneDrive\\Desktop\\bmsce\\heart disease prediction\\knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('C:\\Users\\Quest\\OneDrive\\Desktop\\bmsce\\heart disease prediction\\random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Collect user input
def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex', ('Male', 'Female'))
    chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    cholesterol = st.number_input('Cholesterol in mg/dl', min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('Yes', 'No'))
    resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.0, value=1.0)
    st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Up', 'Flat', 'Down'])

    # Encoding the categorical variables
    sex = 1 if sex == 'Male' else 0
    chest_pain_type = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}[chest_pain_type]
    fasting_bs = 1 if fasting_bs == 'Yes' else 0
    resting_ecg = {'Normal': 0, 'ST': 1, 'LVH': 2}[resting_ecg]
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope]

    # Creating a DataFrame for user input
    user_input = pd.DataFrame(
        [[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    )

    return user_input

# Get user input
user_input_df = user_input_features()
st.write("User Input Features:")
st.write(user_input_df)

# Predict button
if st.button('Predict'):
    # Make predictions using different models
    models = {'Logistic Regression': log_reg_model, 'SVM': svm_model, 'KNN': knn_model, 'Random Forest': rf_model}
    predictions = {}

    for model_name, model in models.items():
        predictions[model_name] = model.predict(user_input_df)[0]

    # Display predictions
    st.write("### Predictions:")
    for model_name, prediction in predictions.items():
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        st.write(f"{model_name}: {result}")


