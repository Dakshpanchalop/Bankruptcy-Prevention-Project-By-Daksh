# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:20:25 2024

@author: prati
"""

import pandas as pd
import streamlit as st 
from sklearn.svm import SVC

st.title('Model Deployment: Bankruptcy using Support vector machine')

st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('industrial_risk',('0','0.5','1'))
    management_risk = st.sidebar.selectbox('management_risk',('0','0.5','1'))
    financial_flexibility = st.sidebar.selectbox('financial_flexibility',('0','0.5','1'))
    credibility = st.sidebar.selectbox('credibility',('0','0.5','1'))
    competitiveness = st.sidebar.selectbox('competitiveness',('0','0.5','1'))
    operating_risk = st.sidebar.selectbox('operating_risk',('0','0.5','1'))
    
    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk,}
    
    
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()

st.subheader('Data Summary')
st.write('1.	industrial_risk: 0=low risk"C:\Games\DATA Sceince\Project\Project 2\P-415.py", 0.5=medium risk, 1=high risk.')
st.write('2.	management_risk: 0=low risk, 0.5=medium risk, 1=high risk.')
st.write('3.	financial flexibility: 0=low flexibility, 0.5=medium flexibility, 1=high flexibility.')
st.write('4.	credibility: 0=low credibility, 0.5=medium credibility, 1=high credibility.')
st.write('5.	competitiveness: 0=low competitiveness, 0.5=medium competitiveness, 1=high competitiveness.')
st.write('6.	operating_risk: 0=low risk, 0.5=medium risk, 1=high risk.')

st.subheader('User Input parameters')
st.write(df)


Bankruptcy = pd.read_excel("C:\\Users\\PC\\Desktop\\ExcelR Project2\\bankruptcy-prevention (1).xlsx")

Bankruptcy["class_yn"] = 1

Bankruptcy.loc[Bankruptcy[' class'] == 'bankruptcy', 'class_yn'] = 0

Bankruptcy.drop(' class', inplace = True, axis =1)
Bankruptcy.head()

X = Bankruptcy.drop(['class_yn'], axis = 1).values
Y = Bankruptcy['class_yn']

model_poly = SVC(kernel = "poly", probability=True)
model_poly.fit(X,Y)

prediction = model_poly.predict(df)
prediction_proba = model_poly.predict_proba(df)

st.subheader('Bankruptcy')
st.write('Yes' if prediction_proba[0][1] < 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)
