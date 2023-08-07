# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:58:54 2023

@author: umroot
"""
import pandas as pd
import numpy as np
import joblib


def preprocess_date(date_val):
    day = date_val.day
    month = date_val.month
    year = date_val.year
    return day, month, year

def preprocess_year(year_val):
    # Calculate the simplified year representation
    first_year = 2012  # Replace this with the actual first year in your dataset
    simplified_year = year_val - first_year
    return simplified_year

def predict_exchange_rate(date_val, model):
    # Convert date_val to day, month, and year
    day, month, year = preprocess_date(date_val)
    simplified_year = preprocess_year(year)

    # Prepare the test input in the required format
    test_input = np.array([[day, month, simplified_year]], dtype=np.float64)

    # Make predictions using the provided model
    output = model.predict_values(test_input)
    predicted_rate = output[0][0]
    return predicted_rate

def apply_error_correction(predicted_rate, error):
    # Apply the error correction
    corrected_rate = predicted_rate + error
    return corrected_rate

if __name__ == "__main__":
    # Load the trained model from the joblib file
    model = joblib.load('surrogate_model_with_year_QP.joblib')

    # Take user input for the date (replace with your desired date)
    test_date = pd.to_datetime('2023-08-03')  # Change the date to your desired date

    # Get the predicted exchange rate using the model
    predicted_rate = predict_exchange_rate(test_date, model)

    # Apply the error correction (replace with your actual error value)
    error = 0.073661194
    corrected_rate = apply_error_correction(predicted_rate, error)

    print(f"Predicted exchange rate for {test_date}: {corrected_rate}")

