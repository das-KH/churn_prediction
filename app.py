import pickle
import streamlit as st
import pandas as pd
from PIL import Image

xgb = 'models/xgb_and_dicVect.bin'
rf = 'models/RF_and_dicVect.bin'
mlp = 'models/mlp_and_dicVect.bin'
 

 
 
def main():
   
    add_selectbox = st.sidebar.selectbox(
    "Which classifier would you like to use for prediction?",
    ("Random Forest", "MLP","XGBoost"))
    st.sidebar.info('This app is created predict Customer Churn')
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Random Forest':
        with open(rf, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    elif add_selectbox == 'MLP':
        with open(mlp, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    else:
        with open(xgb, 'rb') as f_in:
            dv, model = pickle.load(f_in)


    age = st.number_input('Age :', min_value=18, max_value=90, value=18)
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    location = st.selectbox('Location:', ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'])
    sub_len = st.number_input('Subscription_Length_Months :', min_value=0, max_value=24, value=0)
    monthly_bill = st.number_input('Monthly_Bill :', min_value=0, max_value=500, value=0)
    usage = st.number_input('Total_Usage_GB :', min_value=0, max_value=5000, value=0)       

    output= ""
    output_prob = ""
    input_dict={
        "Age": age,
        "Gender": gender ,
        "Location": location,
        "Subscription_Length_Months": sub_len,
        "Monthly_Bill": monthly_bill,
        "Total_Usage_GB": usage}
    if st.button("Predict"):
        X = dv.transform([input_dict])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5
        output_prob = float(y_pred)
        output = bool(churn)
    st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

 
if __name__ == '__main__':
    main()
