import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_multivariate_faformer_dataloader(input_len=168, output_len=24, batch_size=32):
    """
    Get DataLoaders for FAformer multivariate forecasting.
    Focuses on seasonal patterns across multiple zones.
    """
    # Simulated GEFCom multivariate load data (D003)
    np.random.seed(42)
    periods = 8760
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    
    # 3 zones with distinct periodicities
    z1 = 100 + 20 * np.sin(np.linspace(0, periods*2*np.pi/24, periods))
    z2 = 150 + 30 * np.sin(np.linspace(0, periods*2*np.pi/168, periods)) # Weekly
    z3 = 80 + 15 * np.cos(np.linspace(0, periods*2*np.pi/12, periods)) # Semi-daily
    
    data = pd.DataFrame({'ds': dates, 'zone1': z1, 'zone2': z2, 'zone3': z3})
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['zone1', 'zone2', 'zone3']])
    
    def create_faformer_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_faformer_xy(scaled_data[:train_end])
    val_x, val_y = create_faformer_xy(scaled_data[train_end:val_end])
    test_x, test_y = create_faformer_xy(scaled_data[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler
