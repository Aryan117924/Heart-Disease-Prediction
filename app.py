
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost # Required to load XGBClassifier model

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define median values for numerical imputation (from training data)
# These values were calculated from x_train in the notebook.
median_values = {
    "Age": 47.0,
    "Blood Pressure": 135.0,
    "Cholesterol Level": 226.0,
    "BMI": 27.42084729111409,
    "Sleep Hours": 7.00392398555627,
    "Triglyceride Level": 250.0,
    "Fasting Blood Sugar": 120.0,
    "CRP Level": 8.01633513360434,
    "Homocysteine Level": 12.508552733979462
}

# Best threshold determined during model evaluation
best_threshold = 0.40 # This value was obtained from the notebook's evaluation

st.title("Heart Disease Prediction App")
st.write("Enter the patient's information to predict the likelihood of heart disease.")

# --- Input Features --- 
# Numerical Inputs
age = st.slider("Age", 18, 90, int(median_values['Age']))
blood_pressure = st.slider("Blood Pressure", 90, 200, int(median_values['Blood Pressure']))
cholesterol_level = st.slider("Cholesterol Level", 100, 400, int(median_values['Cholesterol Level']))
bmi = st.slider("BMI", 15.0, 40.0, float(f'{median_values["BMI"]:.2f}'))
sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, float(f'{median_values["Sleep Hours"]:.2f}'))
triglyceride_level = st.slider("Triglyceride Level", 50, 500, int(median_values['Triglyceride Level']))
fasting_blood_sugar = st.slider("Fasting Blood Sugar", 70, 200, int(median_values['Fasting Blood Sugar']))
crp_level = st.slider("CRP Level", 0.0, 20.0, float(f'{median_values["CRP Level"]:.2f}'))
homocysteine_level = st.slider("Homocysteine Level", 5.0, 25.0, float(f'{median_values["Homocysteine Level"]:.2f}'))

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
exercise_habits = st.selectbox("Exercise Habits", ["High", "Medium", "Low"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
family_heart_disease = st.selectbox("Family Heart Disease", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
low_hdl_cholesterol = st.selectbox("Low HDL Cholesterol", ["Yes", "No"])
high_ldl_cholesterol = st.selectbox("High LDL Cholesterol", ["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["High", "Medium", "Low"])
stress_level = st.selectbox("Stress Level", ["High", "Medium", "Low"])
sugar_consumption = st.selectbox("Sugar Consumption", ["High", "Medium", "Low"])

# Create a dictionary for the input features
input_data = {
    'Age': age,
    'Blood Pressure': blood_pressure,
    'Cholesterol Level': cholesterol_level,
    'BMI': bmi,
    'Sleep Hours': sleep_hours,
    'Triglyceride Level': triglyceride_level,
    'Fasting Blood Sugar': fasting_blood_sugar,
    'CRP Level': crp_level,
    'Homocysteine Level': homocysteine_level,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Exercise Habits_Low': 1 if exercise_habits == "Low" else 0,
    'Exercise Habits_Medium': 1 if exercise_habits == "Medium" else 0,
    'Smoking_Yes': 1 if smoking == "Yes" else 0,
    'Family Heart Disease_Yes': 1 if family_heart_disease == "Yes" else 0,
    'Diabetes_Yes': 1 if diabetes == "Yes" else 0,
    'High Blood Pressure_Yes': 1 if high_blood_pressure == "Yes" else 0,
    'Low HDL Cholesterol_Yes': 1 if low_hdl_cholesterol == "Yes" else 0,
    'High LDL Cholesterol_Yes': 1 if high_ldl_cholesterol == "Yes" else 0,
    'Alcohol Consumption_Low': 1 if alcohol_consumption == "Low" else 0,
    'Alcohol Consumption_Medium': 1 if alcohol_consumption == "Medium" else 0,
    'Stress Level_Low': 1 if stress_level == "Low" else 0,
    'Stress Level_Medium': 1 if stress_level == "Medium" else 0,
    'Sugar Consumption_Low': 1 if sugar_consumption == "Low" else 0,
    'Sugar Consumption_Medium': 1 if sugar_consumption == "Medium" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply feature engineering exactly as in the notebook
input_df['Age_BP'] = input_df['Age'] * input_df['Blood Pressure']
input_df['BMI_Sugar'] = input_df['BMI'] * input_df['Fasting Blood Sugar']
input_df['Chol_TG'] = input_df['Cholesterol Level'] * input_df['Triglyceride Level']

# Ensure column order matches training data (important for XGBoost)
# This is derived from the 'x' DataFrame's columns in the notebook
expected_columns = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
    'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level',
    'Homocysteine Level', 'Gender_Male', 'Exercise Habits_Low',
    'Exercise Habits_Medium', 'Smoking_Yes', 'Family Heart Disease_Yes',
    'Diabetes_Yes', 'High Blood Pressure_Yes', 'Low HDL Cholesterol_Yes',
    'High LDL Cholesterol_Yes', 'Alcohol Consumption_Low',
    'Alcohol Consumption_Medium', 'Stress Level_Low', 'Stress Level_Medium',
    'Sugar Consumption_Low', 'Sugar Consumption_Medium', 'Age_BP',
    'BMI_Sugar', 'Chol_TG'
]

# Reindex the input_df to match the expected columns, filling missing with 0 for new features (though there shouldn't be any)
input_df = input_df.reindex(columns=expected_columns, fill_value=0)


if st.button("Predict Heart Disease Risk"):
    prediction_proba = model.predict_proba(input_df)[:, 1]
    prediction = (prediction_proba >= best_threshold).astype(int)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease! (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease. (Probability: {prediction_proba[0]:.2f})")

    st.write("\n--- Model Details ---")
    st.write(f"Prediction Threshold: {best_threshold:.2f}")
    st.write("Probability values above this threshold are classified as 'High Risk'.")
