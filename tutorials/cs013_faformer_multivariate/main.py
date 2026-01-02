"""
FAformer Multivariate Forecasting - Main Entry Point
=====================================================
This script provides a CLI interface for FAformer-based multivariate forecasting,
reproducing the results discussed in Chapter 7 of the book.

Usage:
    python main.py --config configs/faformer_multivariate.yaml
"""
import argparse
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FrequencyAttention(nn.Module):
    """Frequency-domain attention mechanism."""
    
    def __init__(self, d_model: int, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Apply FFT along sequence dimension
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Attention in frequency domain
        q = self.query(x_freq.real)
        k = self.key(x_freq.real)
        v = self.value(x_freq.real)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        
        # Inverse FFT
        attended_complex = torch.complex(attended, x_freq.imag[:, :attended.size(1), :])
        output = torch.fft.irfft(attended_complex, n=seq_len, dim=1)
        
        return self.out_proj(output)


class FAformerBlock(nn.Module):
    """FAformer encoder block with frequency attention."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.freq_attention = FrequencyAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.freq_attention(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class FAformerForecaster(nn.Module):
    """FAformer model for multivariate time series forecasting."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, output_size: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            FAformerBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, output_size * horizon)
        self.horizon = horizon
        self.output_size = output_size
    
    def forward(self, x):
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_projection(x[:, -1, :])
        return out.view(-1, self.horizon, self.output_size)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_multivariate_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


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
            torch.save(model.state_dict(), 'best_faformer.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_faformer.pt'))
    return model


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)
    num_features = config.get('num_features', 3)  # e.g., load, price, temperature
    
    time = np.arange(n_samples)
    
    # Simulate correlated multivariate time series
    load = 100 + 20 * np.sin(2 * np.pi * time / 24) + 5 * np.random.randn(n_samples)
    price = 50 + 0.3 * load + 10 * np.random.randn(n_samples)  # Correlated with load
    temperature = 20 + 10 * np.sin(2 * np.pi * time / (24 * 365)) + 2 * np.random.randn(n_samples)
    
    data = np.stack([load, price, temperature], axis=1)
    
    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_normalized = (data - mean) / std
    
    lookback = config.get('lookback', 168)
    horizon = config.get('horizon', 24)
    X, y = create_multivariate_sequences(data_normalized, lookback, horizon)
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = FAformerForecaster(
        input_size=num_features,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        output_size=num_features,
        horizon=horizon,
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Forecasting {num_features} features: load, price, temperature")
    
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
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # Denormalize
    predictions = predictions * std + mean
    actuals = actuals * std + mean
    
    # Calculate per-feature metrics
    feature_names = ['Load', 'Price', 'Temperature']
    print(f"\nTest Results (by feature):")
    for i, name in enumerate(feature_names):
        mae = np.mean(np.abs(actuals[:, :, i] - predictions[:, :, i]))
        rmse = np.sqrt(np.mean((actuals[:, :, i] - predictions[:, :, i]) ** 2))
        print(f"  {name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    
    print(f"\nExpected: Superior performance on correlated multivariate series")
    
    return {'predictions': predictions, 'actuals': actuals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAformer Multivariate Forecasting')
    parser.add_argument('--config', type=str, default='configs/faformer_multivariate.yaml')
    args = parser.parse_args()
    main(args.config)
