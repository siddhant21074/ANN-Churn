import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


model = tf.keras.models.load_model('model.h5')


with open("label_encoder_gender.pkl","rb") as file:
    label_encode = pickle.load(file)
with open("onehot_encoder_gender.pkl","rb") as file:
    onehot_encode = pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler_encode = pickle.load(file)
             

st.title('Customer Churn Prediction')


geography = st.selectbox('Geography',onehot_encode.categories_[0])
gender = st.selectbox('Gender',label_encode.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score =  st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number of product',1,4)
has_cr_card = st.selectbox('Has Credit Card ',[0,1])
is_active_member = st.selectbox('Is Active member ',[0,1])

input_data = pd.DataFrame({
    'CreditScore':credit_score,
    'Gender':[label_encode.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded = onehot_encode.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled = scaler_encode.transform(input_data)

predict = model.predict(input_data_scaled)

prob = predict[0][0]
st.write(f"Churn Probability :{prob}")

if prob > 0.5:
    print("User might churn")
else:
    print("User may not churn")