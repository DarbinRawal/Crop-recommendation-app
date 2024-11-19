import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('Crop_recommendation_model.pkl')

# App title
st.title('🌱 Crop Recommendation System ')

# Sidebar for input parameters
with st.sidebar:
    st.header('Input Parameters')
    N = st.number_input('Nitrogen content in soil (N)', min_value=0.0, value=50.0)
    P = st.number_input('Phosphorus content in soil (P)', min_value=0.0, value=30.0)
    K = st.number_input('Potassium content in soil (K)', min_value=0.0, value=50.0)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input('pH value of the soil', min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=100.0)

# About section
st.markdown("""
### About This App
This application recommends the most suitable crop based on soil and environmental factors like:
- **Soil content**: Nitrogen, Phosphorus, and Potassium.
- **Environmental factors**: Temperature, humidity, and rainfall.
- **Soil pH**: Acidity or alkalinity of the soil.

It uses a machine learning model to make data-driven recommendations. Adjust the inputs in the sidebar and get your recommendation instantly.
""")

# Predict crop
if st.button('🌟 Get Crop Recommendation'):
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    prediction = model.predict(input_data)
    st.success(f'Recommended Crop: {prediction[0]} 🌾')
    st.write("Based on the input conditions, this crop is most suitable for your soil and environment.")


# Visualization
if st.button('Show Predictions with Visualization'):
    # Simulate prediction probabilities for demonstration
    crops = ['Wheat', 'Rice', 'Maize', 'Sugarcane', 'Cotton']
    scores = [0.85, 0.75, 0.65, 0.45, 0.30]  # Example scores (replace with model probabilities)
    
    fig, ax = plt.subplots()
    sns.barplot(x=scores, y=crops, ax=ax, palette='viridis')
    ax.set_title('Crop Suitability Scores')
    ax.set_xlabel('Suitability Score')
    ax.set_ylabel('Crops')
    st.pyplot(fig)
