import pandas as pd
from pathlib import Path

def predict(x):
    # Load saved parameters
    path = Path(__file__).parent.parent / 'model' / 'model_parameters.csv'
    params = pd.read_csv(path)
    slope = params['slope'].iloc[0]
    intercept = params['intercept'].iloc[0]

    # Return prediction
    return slope * x + intercept

