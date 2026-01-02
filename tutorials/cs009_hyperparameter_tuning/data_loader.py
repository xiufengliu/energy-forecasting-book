import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_tuning_dataloader(input_len=24, output_len=1, batch_size=32):
    """
    Get DataLoaders for tuning experiments.
    """
    # Simulated PJM data loading (D001)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    load = 5000 + 1000 * np.sin(np.linspace(0, 365*2*np.pi, 8760)) + np.random.normal(0, 100, 8760)
    data = pd.DataFrame({'ds': dates, 'y': load})
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['y']])
    
    def create_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_xy(scaled_data[:train_end])
    val_x, val_y = create_xy(scaled_data[train_end:val_end])
    test_x, test_y = create_xy(scaled_data[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler
