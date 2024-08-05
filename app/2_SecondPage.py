import streamlit as st
import pandas as pd

model = st.session_state['model']


gender = st.select_slider(label="Gender", options=["Male","Female"])
seniorCitizen = st.select_slider(label = 'SeniorCitizen', options=[0,1])
partner = st.select_slider(label = 'Partner', options=['No','Yes'])
dependents = st.select_slider(label = 'Dependends', options=['No','Yes'])
tenure = st.slider("Tenure",0,75,step=1)
phoneservice = st.select_slider(label = 'Phone Service', options=['No','Yes'])
MultipleLines = st.select_slider(label = 'Multiple Lines', options=['No','Yes'])
InternetService = st.select_slider(label = 'Internet Service', options=['No','DSL','Fiber optic'])
OnlineSecurity = st.selectbox(label = 'Online Security', options=['No','Yes','No internet service'])
OnlineBackup = st.selectbox(label = 'Online Backup', options=['No','Yes','No internet service'])
DeviceProtection = st.selectbox(label = 'Device Protection', options=['No','Yes','No internet service'])
TechSupport = st.selectbox(label = 'Tech Support', options=['No','Yes','No internet service'])
StreamingTV = st.selectbox(label = 'Streaming TV', options=['No','Yes','No internet service'])
StreamingMovies = st.selectbox(label = 'Streaming Movies', options=['No','Yes','No internet service'])
Contract = st.selectbox(label = 'Contract', options=['One year','Two year','Month-to-month'])
PaperlessBilling = st.select_slider(label = 'Paperless Billing', options=['No','Yes'])
PaymentMethod = st.selectbox(label = 'Payment Method', options=['Credit card (automatic)','Mailed check','Electronic check'])
MonthlyCharges = st.slider("Monthly Charges",0,118,step=1)


button = st.button("Confirm")

if button:
    TotalCharges = float(tenure * MonthlyCharges)
    first_month_tenure = 1 if tenure < 3 else 0
    low_monthly_charges = 1 if MonthlyCharges < 30 else 0


    df = pd.DataFrame({
    'gender':gender,
    'SeniorCitizen':seniorCitizen,
    'Partner':partner,
    'Dependents':dependents,
    'tenure':tenure,
    'PhoneService':phoneservice,
    'MultipleLines':MultipleLines,
    'InternetService':InternetService,
    'OnlineSecurity':OnlineSecurity,
    'OnlineBackup':OnlineBackup,
    'DeviceProtection':DeviceProtection,
    'TechSupport':TechSupport,
    'StreamingTV':StreamingTV,
    'StreamingMovies':StreamingMovies,
    'Contract':Contract,
    'PaperlessBilling':PaperlessBilling,
    'PaymentMethod':PaymentMethod,
    'MonthlyCharges':MonthlyCharges,
    'TotalCharges':TotalCharges,
    'first_month_tenure':first_month_tenure,
    'lowMonthlyCharges':low_monthly_charges
    },index=[0])


    prediction = model.predict_proba(df)[:, 1]
    formatted_churn_probability = "{:.2%}".format(prediction.item())

    big_text = f"<h1>Churn Probability: {formatted_churn_probability}</h1>"


    st.markdown(big_text, unsafe_allow_html=True)




