import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model():
    """Load the trained gradient boosting regressor model using joblib"""
    try:
        # Load with joblib (recommended for scikit-learn models)
        model = joblib.load('gradient_boosting_regressor_model.pkl')
        return model
    except Exception as e1:
        st.error("**Model Loading Failed**")


def convert_day_to_category(day_name):
    """Convert day name to Weekday/Weekend category"""
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    if day_name in weekdays:
        return 'Weekday'
    elif day_name in weekends:
        return 'Weekend'
    else:
        return 'Weekday'  # Default fallback

def predict_energy_consumption(model, building_type, square_footage, occupants, appliances, temperature, day_category):
    """Make energy consumption prediction using the trained model with OneHotEncoder"""
    try:
        # Create input dataframe with the same structure as training data
        # The model's preprocessing pipeline will handle OneHotEncoding
        input_data = pd.DataFrame({
            'Building Type': [building_type],
            'Square Footage': [square_footage],
            'Number of Occupants': [occupants],
            'Appliances Used': [appliances],
            'Average Temperature': [temperature],
            'Day of Week': [day_category]
        })
        
        # Make prediction using the dataframe
        # The model pipeline should handle preprocessing including OneHotEncoding
        prediction = model.predict(input_data)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Make sure the model was trained with the same feature names and structure")
        return None

def main():
    # Title and description
    st.title("⚡ Energy Consumption Predictor")
    st.markdown("""
    This application predicts energy consumption based on building characteristics and environmental factors.
    Simply enter the building details below to get an energy consumption prediction.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🏢 Building Information")
        
        # Building Type
        building_type = st.selectbox(
            "Building Type",
            options=['Residential', 'Commercial', 'Industrial'],
            help="Select the type of building"
        )
        
        # Square Footage
        square_footage = st.number_input(
            "Square Footage",
            min_value=100,
            max_value=100000,
            value=25000,
            step=100,
            help="Enter the total square footage of the building"
        )
        
        # Number of Occupants
        occupants = st.number_input(
            "Number of Occupants",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
            help="Enter the number of people occupying the building"
        )
    
    with col2:
        st.header("🏠 Usage & Environment")
        
        # Appliances Used
        appliances = st.number_input(
            "Number of Appliances",
            min_value=1,
            max_value=100,
            value=15,
            step=1,
            help="Enter the total number of appliances in use"
        )
        
        # Average Temperature
        temperature = st.slider(
            "Average Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=30.0,
            step=0.1,
            help="Select the average temperature"
        )
        
        # Day of Week
        day_of_week = st.selectbox(
            "Day of Week",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            help="Select the day of the week"
        )
        
        # Convert day to category
        day_category = convert_day_to_category(day_of_week)
        st.info(f"📅 {day_of_week} → {day_category}")
    
    # Electricity Rate Input
    st.header("💰 Electricity Rate")
    electricity_rate = st.number_input(
        "Electricity Rate (₹ per kWh)",
        min_value=0.1,
        max_value=50.0,
        value=8.0,
        step=0.1,
        help="Enter the electricity rate in Indian Rupees per kWh in your area"
    )
    
    # Prediction section
    st.header("🔮 Energy Consumption Prediction")
    
    if st.button("Predict Energy Consumption", type="primary"):
        with st.spinner("Calculating energy consumption..."):
            prediction = predict_energy_consumption(
                model, building_type, square_footage, occupants, 
                appliances, temperature, day_category
            )
            
            if prediction is not None:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Predicted Energy Consumption",
                        value=f"{prediction:.2f} kWh"
                    )
                
                with col2:
                    # Calculate cost estimation using user-provided rate in INR
                    cost = prediction * electricity_rate
                    st.metric(
                        label="Estimated Cost",
                        value=f"₹{cost:.2f}"
                    )
                
                # Additional insights
                st.subheader("📊 Insights")
                
                insights = []
                
                if building_type == "Commercial":
                    insights.append("🏢 Commercial buildings typically consume more energy due to higher occupancy and equipment usage.")
                elif building_type == "Industrial":
                    insights.append("🏭 Industrial buildings often have the highest energy consumption due to heavy machinery and manufacturing processes.")
                elif building_type == "Residential":
                    insights.append("🏠 Residential buildings generally have lower energy consumption compared to commercial and industrial facilities.")
                
                if day_category == "Weekday":
                    insights.append("📅 Weekday consumption is generally higher due to increased occupancy and activity.")
                
                if temperature > 30:
                    insights.append("🌡️ High temperature increases cooling costs significantly.")
                elif temperature < 18:
                    insights.append("❄️ Low temperature increases heating requirements.")
                
                if occupants > 50:
                    insights.append("👥 High occupancy leads to increased energy demand.")
                
                if appliances > 30:
                    insights.append("🔌 High number of appliances contributes to increased energy consumption.")
                
                for insight in insights:
                    st.info(insight)
    
    # Display sample data
    st.header("📋 Sample Data Reference")
    sample_data = pd.DataFrame({
        'Building Type': ['Residential', 'Commercial', 'Industrial', 'Residential', 'Commercial', 'Industrial'],
        'Square Footage': [24563, 27583, 45313, 41625, 36720, 52000],
        'Number of Occupants': [15, 56, 25, 84, 58, 35],
        'Appliances Used': [4, 23, 44, 17, 47, 65],
        'Average Temperature': [28.52, 23.07, 33.56, 27.39, 17.08, 22.5],
        'Day of Week': ['Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekday', 'Weekday'],
        'Energy Consumption': [2115.57, 3283.80, 4817.83, 4874.30, 4070.59, 6250.75]
    })
    
    st.dataframe(sample_data, width='stretch')

if __name__ == "__main__":
    main()
