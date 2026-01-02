import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_price_pipeline_dataloader(market_name, input_len, output_len, batch_size=32):
    """
    Get DataLoaders for the specialized price forecasting pipeline.
    """
    # Simulated Price data loading (D004)
    data = load_market_price_data(market_name)
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    scaled_y = scaler.fit_transform(data[['y']])
    
    def create_price_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_price_xy(scaled_y[:train_end])
    val_x, val_y = create_price_xy(scaled_y[train_end:val_end])
    test_x, test_y = create_price_xy(scaled_y[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler

def load_market_price_data(market_name):
    """
    Simulated market price data with spikes.
    """
    np.random.seed(42)
    periods = 8760
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    base = 40 + 15 * np.sin(np.linspace(0, periods*2*np.pi/24, periods))
    spikes = np.where(np.random.rand(periods) > 0.99, np.random.uniform(100, 300, periods), 0)
    prices = base + spikes + np.random.normal(0, 5, periods)
    return pd.DataFrame({'ds': dates, 'y': prices})
