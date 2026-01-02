import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_multivariate_load_data():
    """
    Simulated loading of GEFCom2014 multivariate load data (D003).
    Includes multiple zones and weather features.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    
    # 3 zones with different load patterns
    zone1 = 100 + 20 * np.sin(np.linspace(0, 50, 8760)) + np.random.normal(0, 5, 8760)
    zone2 = 150 + 30 * np.sin(np.linspace(0, 50, 8760) + 0.5) + np.random.normal(0, 8, 8760)
    zone3 = 80 + 15 * np.sin(np.linspace(0, 50, 8760) - 0.2) + np.random.normal(0, 3, 8760)
    
    # Temperature feature
    temp = 20 + 10 * np.sin(np.linspace(0, 20, 8760)) + np.random.normal(0, 2, 8760)
    
    data = pd.DataFrame({
        'ds': dates,
        'zone1': zone1,
        'zone2': zone2,
        'zone3': zone3,
        'temp': temp
    })
    return data

def preprocess_multivariate_data(data, target_cols=['zone1', 'zone2', 'zone3'], feature_cols=['temp']):
    """
    Prepare multivariate data for training.
    """
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Scale features and targets separately for easier inverse transformation
    train_x = scaler_x.fit_transform(train_df[feature_cols])
    val_x = scaler_x.transform(val_df[feature_cols])
    test_x = scaler_x.transform(test_df[feature_cols])
    
    train_y = scaler_y.fit_transform(train_df[target_cols])
    val_y = scaler_y.transform(val_df[target_cols])
    test_y = scaler_y.transform(test_df[target_cols])
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (scaler_x, scaler_y)

def create_multivariate_windows(x_data, y_data, input_len, output_len):
    """
    Create sliding windows for multivariate training.
    """
    x_wins, y_wins = [], []
    # Concatenate features and targets for input if needed
    full_input = np.concatenate([y_data, x_data], axis=1)
    
    for i in range(len(y_data) - input_len - output_len + 1):
        x_wins.append(full_input[i:i + input_len])
        y_wins.append(y_data[i + input_len:i + input_len + output_len])
        
    return np.array(x_wins), np.array(y_wins)
