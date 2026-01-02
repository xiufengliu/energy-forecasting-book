import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPipeline:
    """Standard preprocessing pipeline for energy time series."""
    
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.is_fitted = False
        
    def add_calendar_features(self, df, timestamp_col='timestamp'):
        """Extract temporal features from timestamp."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
    
    def create_lags(self, df, target_col, lags):
        """Create lagged features."""
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    def create_sequences(self, data, window_size, horizon=1):
        """Convert time series to windowed sequences (X, y)."""
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:(i + window_size)])
            y.append(data[(i + window_size):(i + window_size + horizon)])
        return np.array(X), np.array(y)
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.1):
        """Temporal split of data."""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        return train, val, test
    
    def fit_transform(self, train_data, cols):
        """Fit scaler on training data and transform."""
        self.scaler.fit(train_data[cols])
        self.is_fitted = True
        return self.scaler.transform(train_data[cols])
    
    def transform(self, data, cols):
        """Apply fitted scaler to data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted on training data before transform.")
        return self.scaler.transform(data[cols])

def handle_missing_values(df, target_col, method='ffill'):
    """Handle missing values in time series."""
    if method == 'ffill':
        return df[target_col].fillna(method='ffill')
    elif method == 'interpolate':
        return df[target_col].interpolate(method='linear')
    return df[target_col]
