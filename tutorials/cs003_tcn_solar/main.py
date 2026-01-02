"""
TCN Solar Forecasting - Main Entry Point
=========================================
This script provides a CLI interface for TCN-based solar forecasting,
reproducing the results discussed in Chapter 5 of the book.

Usage:
    python main.py --config configs/tcn_solar.yaml
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)


class TCNForecaster(nn.Module):
    """TCN model for solar forecasting."""
    
    def __init__(self, input_size: int, num_channels: list, kernel_size: int,
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        out = self.fc(out[:, :, -1])
        return out


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (y_true.max() - y_true.min()) * 100


def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tcn.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_tcn.pt'))
    return model


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)
    time = np.arange(n_samples)
    
    # Simulate solar generation with daily pattern and cloud effects
    hour_of_day = time % 24
    solar_angle = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    cloud_factor = 0.7 + 0.3 * np.random.rand(n_samples)
    seasonal = 0.8 + 0.2 * np.sin(2 * np.pi * time / (24 * 365))
    solar = 50 * solar_angle * cloud_factor * seasonal
    
    mean, std = solar.mean(), solar.std()
    solar_normalized = (solar - mean) / std
    
    lookback = config.get('lookback', 48)
    horizon = config.get('horizon', 24)
    X, y = create_sequences(solar_normalized, lookback, horizon)
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train).unsqueeze(-1), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val).unsqueeze(-1), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test).unsqueeze(-1), torch.FloatTensor(y_test))
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = TCNForecaster(
        input_size=1,
        num_channels=config.get('num_channels', [32, 32, 32]),
        kernel_size=config.get('kernel_size', 3),
        output_size=horizon,
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Training...")
    model = train_model(model, train_loader, val_loader, config, device)
    
    print("\nEvaluating on test set...")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    predictions = np.concatenate(predictions) * std + mean
    actuals = np.concatenate(actuals) * std + mean
    
    nrmse = calculate_nrmse(actuals, predictions)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    print(f"\nTest Results:")
    print(f"  nRMSE: {nrmse:.2f}%")
    print(f"  MAE:   {mae:.2f}")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"\nExpected nRMSE range: 8% - 15%")
    
    return {'nrmse': nrmse, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCN Solar Forecasting')
    parser.add_argument('--config', type=str, default='configs/tcn_solar.yaml')
    args = parser.parse_args()
    main(args.config)
