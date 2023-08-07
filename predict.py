import pandas as pd
from smt.surrogate_models import QP # QP is a second-order polynomial approximaiton 
import joblib # This is used to save the model so we can reuse it later without redoing the training process 

def preprocess_date(date_str):
    year, month, day = date_str.year, date_str.month, date_str.day
    return day, month

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

    # Preprocess the date column to extract day and month
    df['Day'], df['Month'] = zip(*df[date_column_name].apply(preprocess_date))

    # Drop the original date column
    df.drop(columns=[date_column_name], inplace=True)

    # Rename the US_EU_Column to Exchange_Rate
    df.rename(columns={us_eu_column_name: 'Exchange_Rate'}, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # If you want to calculate US_EU rate, uncomment the following line:
    # df['US_Rate'] = df['Exchange_Rate'] * df['EU_Rate']

    print(df)
    
    # Select the input features (day, month,add year if needed)
    X = df.iloc[:, [1,  2]].values
    print(X)
    
    ''' Here we store the dependent variables. This is the first column of the excel sheet i.e. the USD_EUR  '''
    # Select the target variable exchange rate
    y = df.iloc[:, 0].values
    
    print(y)

    ''' Here we initialize the model object using the QP method
    After this we train the model to understand the relation ship between our inputs defined in X and the corresponding 
    outputs defined in y '''
    
    sm = QP()
    sm.set_training_values(X, y)
    sm.train() # This trains the model 
    # Save the model to a file
    joblib.dump(sm, 'surrogate_model_with_QP.joblib')
if __name__ == "__main__":
    main()
