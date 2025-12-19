# Heart-Disease-Prediction
This project is a machine learning–based system that predicts the risk of heart disease using health and lifestyle data. The model is trained on medical features such as age, blood pressure, cholesterol, BMI, and other related factors. 

## 🎯 Problem Statement
- Heart disease datasets are highly imbalanced, with significantly fewer positive cases compared to negative cases.  
- Using accuracy as the primary metric often results in models that miss most at-risk patients.
- The objective of this project is to maximize recall for heart disease cases while maintaining stable and realistic performance.

## 🧠 Approach & Methodology

### 1️⃣ Data Preprocessing
- Handled missing values using median imputation (robust for medical data)
- Applied one-hot encoding for categorical variables
- Converted boolean features into numerical (0/1) format
- Scaled only numeric features using StandardScaler

### 2️⃣ Handling Class Imbalance
- Dataset distribution: approximately 80% No Disease and 20% Disease
- Used class weighting (scale_pos_weight) in XGBoost
- Evaluated model performance using Recall, F1-score, and ROC-AUC instead of accuracy

### 3️⃣ Model Selection
- XGBoost Classifier was used for prediction
- Hyperparameters tuned for better generalization
- Model evaluation based on probability outputs rather than hard predictions

### 4️⃣ Decision Threshold Optimization
- The default prediction threshold (0.5) resulted in very low recall for heart disease cases.
- The decision threshold was optimized using the Precision–Recall trade-off.
- Final selected threshold: 0.40
- This significantly improved detection of positive heart disease cases.

## 📊 Model Performance

### Classification Report (Test Set)

Class 0 (No Disease):
- Precision: 0.80
- Recall: 0.57

Class 1 (Heart Disease):
- Precision: 0.20
- Recall: 0.41
- F1-score: 0.27

Overall Accuracy: 0.54

## ✅ Key Insights
- Recall for heart disease cases improved from ~18% to 41%
- Accuracy was intentionally sacrificed to improve detection of high-risk patients
- The model is suitable for screening and early risk identification
- Accuracy is not the primary metric for this medical prediction task

## 🌐 Real-Time Deployment

The trained model is deployed using Streamlit for real-time heart disease risk prediction.

Features include:
- User-friendly input form
- Probability-based risk score
- Threshold-based classification
- Clear High Risk / Low Risk output

## 🗂 Project Structure

heart_disease_app/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained XGBoost model
├── requirements.txt
└── README.md




