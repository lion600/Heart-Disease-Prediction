import streamlit as st
import numpy as np
import pandas as pd
import pickle
from info import info1
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport



def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# rf_model_result,knn_model_result,svm_model_result,xg_model_result = info1()



knn_model = load_model("knn_model.pkl")
rf_model = load_model("rf_model.pkl")
svm_model = load_model("svm_model.pkl")
xg_model = load_model("xg_model.pkl")



def predict1():

    st.header("Data Profile")
    st.subheader("This page will help you to understand the data")

    input_df = pd.read_csv("heart.csv")
    st.write(input_df.head())
    st.write('\n')
    st.subheader("Understand these variables")
    st.write(
    """



- **Age**: Age of the patient in years. Age is a crucial factor in heart disease risk assessment as the likelihood of developing cardiovascular issues increases with age. 🎂

- **Sex**: Gender of the patient (0 = female, 1 = male). Sex can influence heart disease risk factors and manifestations, with some variations in symptoms and prevalence between males and females. ♂️ ♀️

- **CP (Chest Pain Type)**: This feature represents the type of chest pain experienced by the patient, categorized into four values. The type of chest pain can provide valuable diagnostic information about potential heart conditions. 💔

- **Trestbps (Resting Blood Pressure)**: The resting blood pressure of the patient measured in mm Hg. High blood pressure (hypertension) is a significant risk factor for heart disease and can indicate underlying cardiovascular issues. 🩺

- **Chol (Serum Cholesterol)**: Serum cholesterol levels measured in mg/dl. Elevated cholesterol levels can contribute to the buildup of plaque in the arteries, increasing the risk of heart disease. 🍔

- **FBS (Fasting Blood Sugar)**: Indicates whether the patient's fasting blood sugar level is greater than 120 mg/dl (1 = yes, 0 = no). Elevated fasting blood sugar levels may indicate diabetes or pre-diabetes, which are associated with an increased risk of heart disease. 🍬

- **Restecg (Resting Electrocardiographic Results)**: Represents the results of the resting electrocardiogram (ECG/EKG), categorized into three values. Abnormal ECG findings can signal underlying heart abnormalities or conditions. 📈

- **Thalach (Maximum Heart Rate Achieved)**: Maximum heart rate achieved during exercise. The maximum heart rate achieved can provide insights into cardiovascular fitness and potential heart disease risk. 💓

- **Exang (Exercise Induced Angina)**: Indicates whether the patient experiences exercise-induced angina (1 = yes, 0 = no). Angina during physical exertion can be a symptom of underlying coronary artery disease. 🏃‍♂️

- **Oldpeak (ST Depression Induced by Exercise)**: ST depression induced by exercise relative to rest. ST segment changes on an ECG can indicate myocardial ischemia or reduced blood flow to the heart muscle during exercise. 📉

- **Slope**: Represents the slope of the peak exercise ST segment during stress testing, categorized into three values. The slope of the ST segment can provide additional diagnostic information about myocardial ischemia. 🏔️

- **CA (Number of Major Vessels)**: Number of major vessels (0-3) colored by fluoroscopy. The presence and severity of coronary artery disease can be assessed by the number of major vessels affected. 🚑

- **Thal**: Indicates the results of thallium stress testing, categorized into three values. Abnormal thallium test results can indicate areas of reduced blood flow to the heart muscle, suggesting coronary artery disease. 💉


    """
    )
    
    profile = ProfileReport(input_df)
    st_profile_report(profile)



    
    

    





    

