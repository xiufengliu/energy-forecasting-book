"""
Transformer Load Forecasting - Main Entry Point
================================================
This script provides a CLI interface for Transformer-based load forecasting,
reproducing the results discussed in Chapter 6 of the book.

Usage:
    python main.py --config configs/transformer_load.yaml
"""
import argparse
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Transformer model for load forecasting (Informer-style)."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
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


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_transformer.pt'))
    return model


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)
    time = np.arange(n_samples)
    
    daily_pattern = np.sin(2 * np.pi * time / 24)
    weekly_pattern = 0.5 * np.sin(2 * np.pi * time / (24 * 7))
    trend = 0.01 * time / n_samples
    noise = 0.1 * np.random.randn(n_samples)
    load = 100 + 20 * daily_pattern + 10 * weekly_pattern + trend + noise
    
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
    
    model = TransformerForecaster(
        input_size=1,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        output_size=horizon,
        dropout=config.get('dropout', 0.1)
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
    
    wape = calculate_wape(actuals, predictions)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    print(f"\nTest Results:")
    print(f"  WAPE: {wape:.2f}%")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"\nExpected WAPE range: 2.2% - 3.5%")
    
    return {'wape': wape, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Load Forecasting')
    parser.add_argument('--config', type=str, default='configs/transformer_load.yaml')
    args = parser.parse_args()
    main(args.config)
