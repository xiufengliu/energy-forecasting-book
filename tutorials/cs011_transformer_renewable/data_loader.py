import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_renewable_transformer_dataloader(input_len=48, output_len=24, batch_size=32):
    """
    Get DataLoaders for Transformer renewable forecasting.
    """
    # Simulated NREL Solar data (D005)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    irradiance = np.clip(1000 * np.sin(np.linspace(0, 8760*2*np.pi/24, 8760)), 0, None)
    solar_gen = irradiance * 0.5 + np.random.normal(0, 10, 8760)
    data = pd.DataFrame({'ds': dates, 'y': np.clip(solar_gen, 0, None)})
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    scaled_y = scaler.fit_transform(data[['y']])
    
    def create_transformer_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            # Input sequence
            x.append(data_arr[i:i+input_len])
            # Target sequence
            y.append(data_arr[i+input_len:i+input_len+output_len])
        # Transpose to (seq_len, batch, input_size) for standard PyTorch Transformer
        return torch.FloatTensor(np.array(x)).transpose(0, 1), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_transformer_xy(scaled_y[:train_end])
    val_x, val_y = create_transformer_xy(scaled_y[train_end:val_end])
    test_x, test_y = create_transformer_xy(scaled_y[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x.transpose(0, 1), train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x.transpose(0, 1), val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x.transpose(0, 1), test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler
