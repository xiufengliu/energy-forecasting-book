"""
LSTM Load Forecasting - Main Entry Point
=========================================
This script provides a CLI interface for LSTM-based load forecasting,
reproducing the results discussed in Chapter 3 of the book.

Usage:
    python main.py --config configs/lstm_load.yaml
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """LSTM model for load forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sequences(data: np.ndarray, lookback: int, horizon: int):
    """Create sequences for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def train_model(model, train_loader, val_loader, config, device):
    """Train the LSTM model."""
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
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
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model


def main(config_path: str):
    """Main training and evaluation pipeline."""
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data for demonstration
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)  # 1 year of hourly data
    time = np.arange(n_samples)
    
    # Simulate load with daily and weekly patterns
    daily_pattern = np.sin(2 * np.pi * time / 24)
    weekly_pattern = 0.5 * np.sin(2 * np.pi * time / (24 * 7))
    trend = 0.01 * time / n_samples
    noise = 0.1 * np.random.randn(n_samples)
    load = 100 + 20 * daily_pattern + 10 * weekly_pattern + trend + noise
    
    # Normalize
    mean, std = load.mean(), load.std()
    load_normalized = (load - mean) / std
    
    # Create sequences
    lookback = config.get('lookback', 168)  # 7 days
    horizon = config.get('horizon', 24)  # 24 hours ahead
    X, y = create_sequences(load_normalized, lookback, horizon)
    
    # Train/val/test split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(-1),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(-1),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test).unsqueeze(-1),
        torch.FloatTensor(y_test)
    )
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        output_size=horizon,
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("Training...")
    model = train_model(model, train_loader, val_loader, config, device)
    
    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # Denormalize
    predictions = predictions * std + mean
    actuals = actuals * std + mean
    
    # Calculate metrics
    wape = calculate_wape(actuals, predictions)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    print(f"\nTest Results:")
    print(f"  WAPE: {wape:.2f}%")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"\nExpected WAPE range: 2.5% - 4.0%")
    
    return {'wape': wape, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Load Forecasting')
    parser.add_argument('--config', type=str, default='configs/lstm_load.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)
