import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_wind_data():
    """
    Simulated loading of NREL wind power data.
    In a real scenario, this would load a CSV file from NREL.
    """
    # Generate synthetic NREL-like wind power data (D006)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    
    # Wind power is highly intermittent
    # We use a mix of sine waves and brownian motion for realism
    base = np.abs(np.sin(np.linspace(0, 100, 8760)) + np.random.normal(0, 0.5, 8760))
    wind_power = np.clip(base * 50, 0, 100) # 0-100 MW range
    
    data = pd.DataFrame({'ds': dates, 'y': wind_power})
    return data

def preprocess_data(data, train_ratio=0.7, val_ratio=0.1):
    """
    Split data into train/val/test and scale features.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[['y']])
    val_scaled = scaler.transform(val_data[['y']])
    test_scaled = scaler.transform(test_data[['y']])
    
    return train_scaled, val_scaled, test_scaled, scaler

def create_windows(data, input_len, output_len):
    """
    Create sliding windows for seq2seq training.
    """
    x, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x_win = data[i:i + input_len]
        y_win = data[i + input_len:i + input_len + output_len]
        x.append(x_win)
        y.append(y_win)
        
    return np.array(x), np.array(y)
