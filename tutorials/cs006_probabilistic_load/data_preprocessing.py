import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_pjm_load_data():
    """
    Simulated loading of PJM hourly load data (D001).
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    
    # Load data with strong daily and weekly seasonality
    load = 5000 + 1000 * np.sin(np.linspace(0, 365*2*np.pi, 8760)) + \
           500 * np.sin(np.linspace(0, 365*2*np.pi/24, 8760)) + \
           np.random.normal(0, 200, 8760)
    
    data = pd.DataFrame({'ds': dates, 'y': load})
    return data

def preprocess_probabilistic_data(data, input_len=24, output_len=1):
    """
    Prepare data for probabilistic forecasting.
    """
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]
    
    scaler = StandardScaler()
    train_y = scaler.fit_transform(train_df[['y']])
    val_y = scaler.transform(val_df[['y']])
    test_y = scaler.transform(test_df[['y']])
    
    return train_y, val_y, test_y, scaler

def create_probabilistic_windows(data, input_len, output_len):
    """
    Create sliding windows.
    """
    x, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + output_len])
        
    return np.array(x), np.array(y)
