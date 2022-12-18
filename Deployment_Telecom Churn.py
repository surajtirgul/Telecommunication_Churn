import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

image =Image.open('Telecom.png')
st.image(image, caption='Telecom')

st.title('Telecommunication Churn')
st.sidebar.header('User Input Parameters')
def user_input_features():
    account_length =st.sidebar.slider('Account_Length', 1, 243)
    voice_mail_plan=st.sidebar.radio('Voice_Mail_Plan 📱', [0,1])
    voice_mail_messages=st.sidebar.slider('Voice_Mail_Messages 🗣️📳', 0, 51)
    day_mins=st.sidebar.slider('Day_Mins',0.00,350.80)
    evening_mins=st.sidebar.slider('Evening_Mins🌇',0.00,363.70)
    night_mins=st.sidebar.slider('Night_Mins🌃',23.20,395.00)
    international_mins=st.sidebar.slider('International_Mins 🌎',0.00, 20.00)
    customer_service_calls=st.sidebar.slider('Insert Customer_Service_Calls',0, 9)
    international_plan=st.sidebar.radio('International_Plan', [0, 1])
    day_calls=st.sidebar.slider('Day_Calls 📞',0, 165)
    day_charge=st.sidebar.slider('Day_Charge',0.00, 59.64)
    evening_calls=st.sidebar.slider('Number of Evening_Calls 🌇📞',0 ,170)
    evening_charge=st.sidebar.slider('Evening_Charge🌇 ',0.00, 30.91)
    night_calls=st.sidebar.slider('Number of Night_Calls 🌃📞',33, 175)
    night_charge=st.sidebar.slider('Night_Charges🌃',1.04, 17.77)
    international_calls=st.sidebar.slider('Number of International_Calls 🌍📞',0, 20)
    international_charge=st.sidebar.slider('International_Charges 🌏📞',0.00,5.40)
    total_charge=st.sidebar.slider('Total_Charges',22.93, 96.15)
    data={'account_length':account_length,
         'voice_mail_plan':voice_mail_plan,
         'voice_mail_messages':voice_mail_messages,
         'day_mins':day_mins,
         'evening_mins':evening_mins,
         'night_mins':night_mins,
         'international_mins':international_mins,
         'customer_service_calls':customer_service_calls,
         'international_plan':international_plan,
         'day_calls':day_calls,
         'day_charge':day_charge,
         'evening_calls':evening_calls,
         'evening_charge':evening_charge,
         'night_calls':night_calls,
         'night_charge':night_charge,
         'international_calls':international_calls,
         'international_charge':international_charge,
         'total_charge':total_charge}
    features =pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader('User Input parameters')
st.write(df)

tlc= pd.read_csv("telecommunications_churn.csv")

X = tlc.iloc[:,0:18]
Y = tlc['churn']

model=XGBRFClassifier()
model.fit(X,Y)

prediction=model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'Customer will not Churn')



