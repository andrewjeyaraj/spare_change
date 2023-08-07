# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:56:45 2023

@author: umroot
"""

import pandas as pd 
import numpy as np
from smt.surrogate_models import QP
import joblib

# Load the saved model from file
loaded_model = joblib.load('surrogate_model_with_year.joblib')

test_input = [4,8,11]  # Expected density is 65.04632

# Convert test input to a 2D array
test_input_array = np.array([test_input])

# Make predictions using the loaded model
output = loaded_model.predict_values(test_input_array)

print("Predicted Rate:", output)