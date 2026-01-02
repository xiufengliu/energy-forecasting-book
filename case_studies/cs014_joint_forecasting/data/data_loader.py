import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_joint_dataloader(batch_size=32, input_len=168, output_len=24):
    """
    Synchronize Load, Solar, and Price data streams.
    """
    periods = 8760
    np.random.seed(42)
    
    # 1. Generate/Load Load Data
    load = 5000 + 1000 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)) + np.random.normal(0, 100, periods)
    
    # 2. Generate/Load Solar Data
    solar = np.clip(500 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)), 0, None) + np.random.normal(0, 20, periods)
    
    # 3. Generate/Load Price Data
    price = 30 + 10 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)) + np.random.normal(0, 5, periods)
    
    df = pd.DataFrame({'load': load, 'solar': solar, 'price': price})
    
    # Scaling
    scalers = {
        'load': StandardScaler(),
        'solar': StandardScaler(),
        'price': StandardScaler()
    }
    
    scaled_data = np.zeros_like(df.values)
    scaled_data[:, 0] = scalers['load'].fit_transform(df[['load']]).flatten()
    scaled_data[:, 1] = scalers['solar'].fit_transform(df[['solar']]).flatten()
    scaled_data[:, 2] = scalers['price'].fit_transform(df[['price']]).flatten()
    
    def create_joint_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len]) # [load, solar, price] history
            y.append(data_arr[i+input_len:i+input_len+output_len]) # [load, solar, price] future
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    n = len(scaled_data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_x, train_y = create_joint_xy(scaled_data[:train_end])
    val_x, val_y = create_joint_xy(scaled_data[train_end:val_end])
    test_x, test_y = create_joint_xy(scaled_data[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scalers
