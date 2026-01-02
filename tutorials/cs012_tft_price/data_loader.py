import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_price_tft_dataloader(input_len=168, output_len=24, batch_size=32):
    """
    Get DataLoaders for TFT price forecasting.
    Includes multiple zones and external features (load, weather).
    """
    # Simulated GEFCom price data (D004)
    np.random.seed(42)
    periods = 8760
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    
    # Prices characterized by spikes and daily cycles
    base_price = 30 + 10 * np.sin(np.linspace(0, periods*2*np.pi/24, periods))
    spikes = np.where(np.random.rand(periods) > 0.98, np.random.uniform(50, 200, periods), 0)
    prices = base_price + spikes + np.random.normal(0, 5, periods)
    
    # External features
    load = 100 + 20 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)) + np.random.normal(0, 5, periods)
    temp = 20 + 5 * np.cos(np.linspace(0, 10, periods)) + np.random.normal(0, 2, periods)
    
    data = pd.DataFrame({'ds': dates, 'price': prices, 'load': load, 'temp': temp})
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['price', 'load', 'temp']])
    
    def create_tft_xy(data_arr):
        x, y = [], []
        # TFT expects (batch, seq_len, num_vars, input_size)
        # For simplicity in this tutorial, we format as (batch, seq_len, num_vars)
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len, 0:1]) # Predict price
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_tft_xy(scaled_data[:train_end])
    val_x, val_y = create_tft_xy(scaled_data[train_end:val_end])
    test_x, test_y = create_tft_xy(scaled_data[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler
