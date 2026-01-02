import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_price_value_dataloader(batch_size=32, input_len=168, output_len=1):
    """
    Get DataLoaders for Value-Oriented price forecasting.
    """
    periods = 8760
    np.random.seed(42)
    
    # 1. Price Data with volatility
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    prices = 40 + 20 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)) + np.random.normal(0, 10, periods)
    df = pd.DataFrame({'ds': dates, 'price': prices})
    
    # Scaling
    scaler = StandardScaler()
    scaled_y = scaler.fit_transform(df[['price']])
    
    def create_value_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    n = len(scaled_y)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_x, train_y = create_value_xy(scaled_y[:train_end])
    val_x, val_y = create_value_xy(scaled_y[train_end:val_end])
    test_x, test_y = create_value_xy(scaled_y[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler
