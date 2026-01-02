import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_renewable_dataloader(source_type, input_len, output_len, batch_size=32):
    """
    Get PyTorch DataLoaders for renewable forecasting.
    Includes NWP features (e.g., irradiance for solar, wind speed for wind).
    """
    if source_type == "solar":
        data = load_solar_nwp_data()
    else:
        data = load_wind_nwp_data()
        
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    # Scale targets and NWP features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['y', 'nwp_feature']])
    
    def create_renewable_xy(data_arr):
        x, y = [], []
        for i in range(len(data_arr) - input_len - output_len + 1):
            # Input includes both past generation and NWP forecast
            x.append(data_arr[i:i+input_len])
            # Target is future generation
            y.append(data_arr[i+input_len:i+input_len+output_len, 0:1])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))
    
    train_x, train_y = create_renewable_xy(scaled_data[:train_end])
    val_x, val_y = create_renewable_xy(scaled_data[train_end:val_end])
    test_x, test_y = create_renewable_xy(scaled_data[val_end:])
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler

def load_solar_nwp_data():
    """
    Simulated solar generation with NWP (irradiance).
    """
    np.random.seed(42)
    periods = 8760
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    irradiance = np.clip(1000 * np.sin(np.linspace(0, periods*2*np.pi/24, periods)), 0, None) + np.random.normal(0, 50, periods)
    generation = 0.8 * irradiance + np.random.normal(0, 20, periods)
    return pd.DataFrame({'ds': dates, 'y': np.clip(generation, 0, None), 'nwp_feature': irradiance})

def load_wind_nwp_data():
    """
    Simulated wind generation with NWP (wind speed).
    """
    np.random.seed(42)
    periods = 8760
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    wind_speed = 10 + 5 * np.sin(np.linspace(0, 100, periods)) + np.random.normal(0, 2, periods)
    generation = 2 * wind_speed**1.5 + np.random.normal(0, 10, periods)
    return pd.DataFrame({'ds': dates, 'y': np.clip(generation, 0, None), 'nwp_feature': wind_speed})
