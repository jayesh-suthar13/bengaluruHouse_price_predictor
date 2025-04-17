import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("RidgeModel.pkl", "rb"))

#Correctly extract OneHotEncoder from named pipeline
try:
    encoder = model.named_steps['columntransformer'].named_transformers_['onehotencoder']
    location_list = sorted(encoder.categories_[0].tolist())
except Exception as e:
    st.error(f"Error loading location list: {e}")
    location_list = []

st.title("🏠 Bengaluru House Price Prediction")

location = st.selectbox("Select Location", location_list)
BHK = st.number_input("Enter BHK (Bedrooms)", min_value=1, step=1)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, step=1)
total_sqft = st.number_input("Enter Total Square Feet", min_value=500, step=50)

def preprocess_inputs(location, BHK, bath, total_sqft):
    return pd.DataFrame([{
        'location': location,
        'total_sqft': total_sqft,
        'bath': bath,
        'BHK': BHK
    }])

if st.button("Predict Price"):
    input_df = preprocess_inputs(location, BHK, bath, total_sqft)
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 Predicted Price: ₹ {prediction:,.2f} Lakhs")
