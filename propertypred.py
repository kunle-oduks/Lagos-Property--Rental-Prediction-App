import streamlit as st

import pandas as pd
import joblib
import numpy as np
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb


@st.cache_data()
def load_data(df):
    data = pd.read_csv(df)
    return data

data = load_data('propertycenter.csv')
data['Estate'] = data['address'].apply(lambda x: x.split()[-3] +' ' + x.split()[-4] if len(x.split()) >3 else '')
data['ppty_type'] =data['desc'].apply(lambda x: x.split()[2] + ' ' + x.split()[3] if len(x.split()) >4 else '') 
data['ppty_type'] = data['ppty_type'].str.replace('/', '')
data['ppty_type'] = data['ppty_type'].str.replace('for', '')

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 60px; font-family: Helvetica'>LAGOS HOME RENT PREDICTION APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Kunle Odukoya Data Science</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

box1, box2, box3, box4, box5 = st.columns(5)
box2.image('pngwing.com (18).png', width = 400,  caption = 'Home Rental Prediction Project')

st.header('Project Background Information',divider = True)
st.write("Home rental is an everyday business in the ever busy Lagos. This app helps a prospective tenant to have a fair idea of the homes in different areas of Lagos as at May 2024. A landlord or a property developer can also use the app to predict their rental incomes. In developing the app, the features and rental value of data of over 21000 Lagos properties were scraped from the website of a popular online property agent in Nigeria. Please note that the app is for academic purposes only. Areas of improvement for the project will include: 1). There is need to get data over a period of years so that the effect of time and inflation can be built into the model. 2)Other home features such as (condition of the property, either newly built or old house) are not stated on the scraped website. This can be included into the model because it has a significant effect on rental value ")


st.sidebar.image('pngwing.com (19).png', width = 300)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

#Input Fetures to collect users data
st.sidebar.write("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Please choose your home features</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
Bedrooms = st.sidebar.number_input('Number of Bedrooms', max_value = 5, min_value = 1)
Bathrooms = st.sidebar.number_input('Number of Bathrooms', max_value = 5, min_value = 1)
Toilets = st.sidebar.number_input('Number of Toilets', max_value = 5, min_value = 1)
Property_type = st.sidebar.selectbox('Property Type', options = data['ppty_type'].unique())
Location = st.sidebar.selectbox('Location', options = data['location'].unique())
Area = st.sidebar.selectbox('Area in the Location chosen', options = data['Estate'].unique())

#dataframe to display users inputs
df = pd.DataFrame()
df['bedrooms'] = [Bedrooms]
df['bathrooms'] = [Bathrooms]
df['toilets'] = [Toilets]
df['location'] = [Location]
df['Estate'] = [Area]
df['ppty_type'] = [Property_type]


st.markdown("<br>", unsafe_allow_html=True)

st.header('User Input', divider = True)
st.dataframe(df, use_container_width = True)

#download saved joblib models to use to transform users data where applicable
model_location = joblib.load('location.encoder.pkl')
model_area = joblib.load('Estate.encoder.pkl')
model_type = joblib.load('ppty_type.encoder.pkl')
model_rfn = joblib.load('model_rfn.pkl')

df['ppty_type'] = model_type.transform(df['ppty_type'])
df['location'] = model_location.transform(df['location'])
df['Estate'] = model_area.transform(df['Estate'])


st.markdown("<br>", unsafe_allow_html=True)
st.header('Converted Input', divider = True)
st.dataframe(df, use_container_width = True)

boxA, boxB, boxC = st.columns(3)

#function for the predict button..ITs called when the user clicks on predict button
def predict():
    predicted_price = model_rfn.predict(df).round(2)
    boxB.markdown(f"<h1 style = 'color: #OOOOOO; text-align: center; font-size: 20px; font-family: Helvetica'>The average predicted price for a {Property_type} with {Bedrooms} bedrooms, {Bathrooms} bathrooms and {Toilets} toilets in {Area}, area of {Location} is N{int(predicted_price):,.2f}</h1>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

boxB.button('PREDICT', on_click = predict)

