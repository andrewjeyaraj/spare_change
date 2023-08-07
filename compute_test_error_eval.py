# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:30:33 2023

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

# ... (other functions same as before)

def run_test_campaign(test_file_path, test_sheet_name, date_column_name, us_eu_column_name, models):
    # Read the test data from the Excel sheet
    test_df = pd.read_excel(test_file_path, sheet_name=test_sheet_name, usecols=[date_column_name, us_eu_column_name])

    # Preprocess the test date column to extract day, month, and year
    test_df['Day'], test_df['Month'], test_df['Year'] = zip(*test_df[date_column_name].apply(preprocess_date))

    # Preprocess the test year column to get the simplified representation
    test_df['Simplified_Year'] = test_df['Year'].apply(preprocess_year)

    # Drop the original date and year columns
    test_df.drop(columns=[date_column_name, 'Year'], inplace=True)

    # Drop rows with NaN values
    test_df.dropna(inplace=True)

    # Initialize a dictionary to store the results for each model
    results = {model_name: [] for model_name in models}

    # Loop through each data point in the test DataFrame and make predictions for each model
    for _, row in test_df.iterrows():
        test_input = [row['Day'], row['Month'], row['Simplified_Year']]
        for model_name, model in models.items():
            output = test_model(model, test_input)
            results[model_name].append(output[0][0])

    # Add the predicted results to the test DataFrame
    for model_name in models:
        test_df[f'Predicted_{model_name}'] = results[model_name]

        # Calculate the error for each predicted point compared to the data point
        test_df[f'Error_{model_name}'] = np.abs(test_df[f'Predicted_{model_name}'] - test_df[us_eu_column_name])

    # Write the results back to the same Excel sheet
    with pd.ExcelWriter(test_file_path, engine='openpyxl', mode='a') as writer:
        test_df.to_excel(writer, sheet_name='Results', index=False)


if __name__ == "__main__":
    file_path = 'euusd.xls'
    sheet_name = 'Daily_1'
    date_column_name = 'observation_date'
    us_eu_column_name = 'selected'

    # Build the models (you can change the model_type and options accordingly)
    models = {
        'RBF': build_model(file_path, sheet_name, date_column_name, us_eu_column_name, model_type='RBF', d0=5),
        'IDW': build_model(file_path, sheet_name, date_column_name, us_eu_column_name, model_type='IDW', p=2),
        'QP': build_model(file_path, sheet_name, date_column_name, us_eu_column_name, model_type='IDW', p=2)
        # Add more models if needed
    }

    # Run the test campaign using the provided test data and models
    test_file_path = 'test_sheet.xlsx'
    test_sheet_name = 'Sheet1'
    run_test_campaign(test_file_path, test_sheet_name, date_column_name, us_eu_column_name, models)
