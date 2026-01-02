import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_dataloader(dataset_name, input_len, output_len, batch_size=32):
    """
    Get PyTorch DataLoaders for the specified dataset.
    """
    # Simulated ERCOT data loading (D002)
    data = load_ercot_data()
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    scaler = StandardScaler()
    train_y = scaler.fit_transform(data.iloc[:train_end][['y']])
    val_y = scaler.transform(data.iloc[train_end:val_end][['y']])
    test_y = scaler.transform(data.iloc[val_end:][['y']])
    
    def create_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            x.append(data_arr[i:i+input_len])
            y.append(data_arr[i+input_len:i+input_len+output_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y_t = create_xy(train_y)
    val_x, val_y_t = create_xy(val_y)
    test_x, test_y_t = create_xy(test_y)
    
    train_loader = DataLoader(TensorDataset(train_x, train_y_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y_t), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y_t), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler

def load_ercot_data():
    """
    Simulated ERCOT load data.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    load = 10000 + 2000 * np.sin(np.linspace(0, 365*2*np.pi, 8760)) + np.random.normal(0, 500, 8760)
    return pd.DataFrame({'ds': dates, 'y': load})
