import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from predict import predict
# from streamlit_extras.switch_page_button import switch_page 
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier





# @st.experimental_memo
# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64("image.jpg")

# page_bg_img = f"""
# <style>

# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://img.freepik.com/free-vector/modern-blue-medical-background_1055-6880.jpg?w=826");
# background-size: 100%;
# backdrop-filter: blur(100px); 
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
# }}


# </style>
# """
# Function to process input data
def process_data(X):
    scaler = StandardScaler()
    input_data_df = pd.DataFrame(data= X)
    X_scaled = scaler.fit_transform(input_data_df)
    return X_scaled


def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

knn_model = load_model("knn_model.pkl")
rf_model = load_model("rf_model.pkl")
svm_model = load_model("svm_model.pkl")
xg_model = load_model("xg_model.pkl")


# main function
def info1():


    # st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Heart Disease Prediction:heart:")

    global rf_model_result 
    rf_model_result = None
    global knn_model_result
    global svm_model_result
    global xg_model_result
    



    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Enter your age", min_value=0)
        sex = st.radio("Select your sex", ["Male", "Female"])
        if sex == 'Male':
            sex = 0
        else:
            sex = 1    

        # combined_prediction = 1    

        cp = st.selectbox("Enter chest pain type", ["typical angina", "atypical angina", "non-anginal pain","asymptomatic"])
        if cp == "typical angina":
            cp = 0
        elif cp == "atypical angina":
            cp = 1
        elif cp == "non-anginal pain":
            cp = 2 
        else:
            cp = 3      

        trestbps = st.number_input("Blood presure")

        chol = st.number_input("cholestoral")

        fbs = st.radio("Fasting blood sugar level above 125 mg/dL",["yes","no"])
        if fbs == "yes":
            fbs = 1
        else:
            fbs = 0    

        
        

    


    with col2:

        restecg = st.selectbox("electrocardiographic results",["normal","having ST-T","hypertrophy"])
        if restecg == "normal":
            restecg = 0
        elif restecg == "having ST-T":
            restecg = 1
        else:
            restecg = 2        
        thalach = st.number_input("Maximun heart rate achived")
        exang = st.radio("exercise induced angina ",["yes","No"])
        if exang == "yes":
            exang = 1
        else:
            exang = 0    

        oldpeak = st.number_input("ST depression induced by exercise relative to rest")

        slope = st.selectbox("the slope of the peak exercise ST segment ",["upsloping","flat","downsloping"])
        if slope == "upsloping":
            slope = 0
        elif slope == "flat":
            slope = 1
        else:
            slope = 2        

        ca  = st.selectbox("no_of_vessels",["No major coronary artery","One major coronary artery"," Two major coronary arteries","Three major coronary arteries"])
        if ca == "No major coronary artery":
            ca = 0
        elif ca ==  "One major coronary artery":
            ca = 1
        elif ca == "Two major coronary arteries":
            ca = 2
        else:
            ca = 3          
        thal = st.radio("thallium-201 ",["normal","fixed defected","reversable defected"])
        if thal == "normal":
            thal = 1
        elif thal == "fixed defected":
            thal = 2
        else:
            thal= 3        

    
    X = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    
    ok = st.button("predict")
    

    if ok:
        X_processed = process_data(X)
        # X_new = np.array(X_processed)
        X_new = X_processed.reshape(-1, 13)
         
        rf_model_result = rf_model.predict(X_new)
        knn_model_result = knn_model.predict(X_new)
        svm_model_result = svm_model.predict(X_new)
        xg_model_result = xg_model.predict(X_new)
        # print(rf_model_result)
        # print(knn_model_result)
        # print(svm_model_result)
        # print(xg_model_result)


        ensemble_result = []


        for rf_pred, knn_pred, svm_pred, xg_pred in zip(rf_model_result, knn_model_result, svm_model_result, xg_model_result):
        # Perform majority voting
            combined_prediction = max(set([rf_pred, knn_pred, svm_pred, xg_pred]), key = [rf_pred, knn_pred, svm_pred, xg_pred].count)
            ensemble_result.append(combined_prediction)

        if combined_prediction == 1:
            st.markdown("**You have heart disease**", unsafe_allow_html=True)

        else:
            st.markdown("**You do not have heart disease**", unsafe_allow_html=True)
   

    
        c1,c2  = st.columns([3,1])

        with c1:
            st.info("Random Forest")
            st.info("K-Nearest Neighbor(KNN) ")
            st.info("Support Vector Machines")
            st.info("XGBoost")
        with c2:
            if rf_model_result == 1:
                st.info("YES")
            else:
                st.info("NO")

            if knn_model_result == 1:
                st.info("YES")
            else:
                st.info("NO")

            if knn_model_result == 1:
                st.info("YES")
            else:
                st.info("NO")

            if xg_model_result == 1:
                st.info("YES")
            else:
                st.info("NO")                 


            


            

    # return rf_model_result,knn_model_result,svm_model_result,xg_model_result
  
       





        

        


    

    
    


    



        
        
            
            

            



    

