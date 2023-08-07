import pandas as pd
from smt.surrogate_models import QP # QP is a second-order polynomial approximaiton 
import joblib # This is used to save the model so we can reuse it later without redoing the training process 
from smt.surrogate_models import RBF
import numpy as np
from smt.surrogate_models import IDW

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

def main():
    # Replace 'file.xlsx' with the path to your Excel file
    file_path = 'euusd.xls'

    # Replace 'Sheet1' with the name of the specific sheet where your data resides
    sheet_name = 'Daily_1'

    # Replace 'DateColumn' and 'US_EU_Column' with the actual column names containing the date and exchange rate respectively
    date_column_name = 'observation_date'
    us_eu_column_name = 'selected'

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

    # If you want to calculate US_EU rate, uncomment the following line:
    # df['US_Rate'] = df['Exchange_Rate'] * df['EU_Rate']


    print(df)
    
    # Select the input features (temperature, pressure, mixture ratio)
    X = df.iloc[:, [1,  2,3]].values.astype(np.float64)
    print(X)
    
    ''' Here we store the dependent variables. This is the first column of the excel sheet i.e. the USD_EUR  '''
    # Select the target variable exchange rate
    y = df.iloc[:, 0].values.astype(np.float64)
    
    print(y)
    
    ''' Here we initialize the model object using the QP method
    After this we train the model to understand the relation ship between our inputs defined in X and the corresponding 
    outputs defined in y '''
    
    sm = IDW(p=2)
    sm.set_training_values(X, y)
    sm.train() # This trains the model 
    # Save the model to a file
    # Save the trained model to a file without pickling the RBF model
    #joblib.dump(sm, 'surrogate_model_with_year_RBF.joblib', compress=True)
    print(df)
    test_input = [4,8,11]  # Expected density is 65.04632

    # Convert test input to a 2D array
    test_input_array = np.array([test_input], dtype=np.float64)

    # Make predictions using the loaded model
    output = sm.predict_values(test_input_array)
    print(output)

if __name__ == "__main__":
    main()
