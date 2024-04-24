import streamlit as st
from predict import predict1
from info import info1



page = st.sidebar.selectbox("predict or info",("predict","Information"))

if page == "predict":
    predict1()
else:
    info1()    


    


