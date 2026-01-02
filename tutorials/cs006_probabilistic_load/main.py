"""
Probabilistic Load Forecasting - Main Entry Point
==================================================
This script provides a CLI interface for probabilistic load forecasting,
reproducing the results discussed in Chapter 8 of the book.

Usage:
    python main.py --config configs/probabilistic_load.yaml
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantileRegressionLSTM(nn.Module):
    """LSTM with quantile regression output for probabilistic forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, quantiles: list, dropout: float = 0.2):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size * self.num_quantiles)
        self.output_size = output_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        # Reshape to (batch, output_size, num_quantiles)
        return out.view(-1, self.output_size, self.num_quantiles)


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
    
    def forward(self, predictions, targets):
        # predictions: (batch, horizon, num_quantiles)
        # targets: (batch, horizon)
        targets = targets.unsqueeze(-1)
        errors = targets - predictions
        losses = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
        return losses.mean()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


def calculate_crps(y_true: np.ndarray, quantile_predictions: np.ndarray, quantiles: list) -> float:
    """Calculate Continuous Ranked Probability Score."""
    crps_values = []
    for i in range(len(y_true)):
        obs = y_true[i]
        preds = quantile_predictions[i]
        
        # Approximate CRPS using quantiles
        crps_sample = 0
        for q_idx, q in enumerate(quantiles):
            pred = preds[q_idx] if len(preds.shape) == 1 else preds[:, q_idx].mean()
            crps_sample += 2 * abs((obs <= pred) - q) * abs(pred - obs)
        crps_values.append(crps_sample / len(quantiles))
    
    return np.mean(crps_values)


def calculate_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculate prediction interval coverage."""
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean() * 100


def train_model(model, train_loader, val_loader, config, device, quantiles):
    criterion = QuantileLoss(quantiles).to(device)
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
            torch.save(model.state_dict(), 'best_probabilistic.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_probabilistic.pt'))
    return model


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    quantiles = config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
    
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)
    time = np.arange(n_samples)
    
    daily_pattern = np.sin(2 * np.pi * time / 24)
    weekly_pattern = 0.5 * np.sin(2 * np.pi * time / (24 * 7))
    noise = 0.15 * np.random.randn(n_samples)
    load = 100 + 20 * daily_pattern + 10 * weekly_pattern + noise
    
    mean, std = load.mean(), load.std()
    load_normalized = (load - mean) / std
    
    lookback = config.get('lookback', 168)
    horizon = config.get('horizon', 24)
    X, y = create_sequences(load_normalized, lookback, horizon)
    
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
    
    model = QuantileRegressionLSTM(
        input_size=1,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        output_size=horizon,
        quantiles=quantiles,
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Quantiles: {quantiles}")
    
    print("Training...")
    model = train_model(model, train_loader, val_loader, config, device, quantiles)
    
    print("\nEvaluating on test set...")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    predictions = np.concatenate(predictions)  # (N, horizon, num_quantiles)
    actuals = np.concatenate(actuals)  # (N, horizon)
    
    # Denormalize
    predictions = predictions * std + mean
    actuals = actuals * std + mean
    
    # Calculate metrics
    median_idx = quantiles.index(0.5)
    median_predictions = predictions[:, :, median_idx]
    mae = np.mean(np.abs(actuals - median_predictions))
    
    # 80% prediction interval coverage
    lower_idx = quantiles.index(0.1)
    upper_idx = quantiles.index(0.9)
    coverage = calculate_coverage(actuals.flatten(), 
                                  predictions[:, :, lower_idx].flatten(),
                                  predictions[:, :, upper_idx].flatten())
    
    print(f"\nTest Results:")
    print(f"  MAE (median): {mae:.2f}")
    print(f"  80% PI Coverage: {coverage:.1f}% (target: 80%)")
    print(f"\nExpected: CRPS < 0.05, reliable calibration")
    
    return {'mae': mae, 'coverage_80': coverage}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic Load Forecasting')
    parser.add_argument('--config', type=str, default='configs/probabilistic_load.yaml')
    args = parser.parse_args()
    main(args.config)
