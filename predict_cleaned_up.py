# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:26:00 2023

@author: umroot
"""

import pandas as pd
import numpy as np
from smt.surrogate_models import QP, RBF, IDW
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

def build_model(file_path, sheet_name, date_column_name, us_eu_column_name, model_type='RBF', **model_options):
    # Read the specific sheet of the Excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[date_column_name, us_eu_column_name])

    # Preprocess the date column to extract day, month, and year
    df['Day'], df['Month'], df['Year'] = zip(*df[date_column_name].apply(preprocess_date))

    # Preprocess the year column to get the simplified representation
    df['Simplified_Year'] = df['Year'].apply(preprocess_year)

    # Drop the original date and year columns
    df.drop(columns=[date_column_name, 'Year'], inplace=True)

    # Rename the US_EU_Column to Exchange_Rate
    df.rename(columns={us_eu_column_name: 'Exchange_Rate'}, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Select the input features (Day, Month, Simplified_Year)
    X = df[['Day', 'Month', 'Simplified_Year']].values.astype(np.float64)

    # Select the target variable (Exchange_Rate)
    y = df['Exchange_Rate'].values.astype(np.float64)

    # Choose the model type and initialize the model
    if model_type == 'QP':
        sm = QP(**model_options)
    elif model_type == 'RBF':
        sm = RBF(**model_options)
    elif model_type == 'IDW':
        sm = IDW(**model_options)
    else:
        raise ValueError("Invalid model_type. Please choose from 'QP', 'RBF', or 'IDW'.")

    # Set the training values and train the model
    sm.set_training_values(X, y)
    sm.train()

    return sm

def test_model(model, test_input):
    # Convert test input to a 2D array and set its dtype to float64
    test_input_array = np.array([test_input], dtype=np.float64)

    # Make predictions using the loaded model
    output = model.predict_values(test_input_array)
    return output
if __name__ == "__main__":
    file_path = 'euusd.xls'
    sheet_name = 'Daily_1'
    date_column_name = 'observation_date'
    us_eu_column_name = 'selected'

    # Build the RBF model (you can change the model_type and options accordingly)
    model = build_model(file_path, sheet_name, date_column_name, us_eu_column_name, model_type='QP')
    joblib.dump(model, 'surrogate_model_with_QP.joblib')

    # Test the model with a test_input
    test_input = [4, 8, 11]  # Expected density is 65.04632
    output = test_model(model, test_input)
    print("Predicted Output:", output)
