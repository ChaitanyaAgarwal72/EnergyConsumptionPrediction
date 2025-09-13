# Energy Consumption Predictor

A Streamlit-based web application that predicts energy consumption for buildings using machine learning.

## Features

- **User-friendly Interface**: Easy-to-use web interface built with Streamlit
- **Smart Day Conversion**: Automatically converts Monday-Sunday input to Weekday/Weekend for the model
- **Real-time Predictions**: Get instant energy consumption predictions
- **Cost Estimation**: Calculates estimated energy costs
- **Efficiency Rating**: Provides building efficiency ratings
- **Insights**: Offers personalized insights based on input parameters

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

3. Fill in the building information:
   - **Building Type**: Select between Residential or Commercial
   - **Square Footage**: Enter the total square footage
   - **Number of Occupants**: Enter the number of people
   - **Number of Appliances**: Enter the total number of appliances
   - **Average Temperature**: Set the average temperature in Celsius
   - **Day of Week**: Select any day from Monday to Sunday

4. Click "Predict Energy Consumption" to get your prediction

## Day of Week Conversion

The application automatically converts day selections:
- **Monday to Friday** → Weekday
- **Saturday to Sunday** → Weekend

This conversion happens automatically in the background, so the model receives the correct format.

## Model Information

The application uses a pre-trained Gradient Boosting Regressor model that was trained on building energy consumption data with the following features:
- Building Type
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Day of Week (Weekday/Weekend)

## Output

The application provides:
- **Energy Consumption Prediction** (in kWh)
- **Estimated Cost** (assuming $0.12 per kWh)
- **Efficiency Rating** (Excellent/Good/Average/Poor)
- **Personalized Insights** based on your input parameters

## Sample Data

The application includes sample data to help you understand the expected input ranges and formats.

## File Structure

```
.
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── gradient_boosting_regressor_model.pkl    # Trained ML model
└── README.md                                # This file
```

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed correctly
2. Ensure the model file (.pkl) is in the same directory as app.py
3. Check that you're using Python 3.8 or higher
4. Verify that Streamlit is properly installed

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn (Gradient Boosting Regressor)
- **Data Processing**: Pandas, NumPy
