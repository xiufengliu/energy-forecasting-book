"""
Hyperparameter Tuning - Main Entry Point
=========================================
This script provides a CLI interface for hyperparameter tuning,
reproducing the results discussed in Chapter 11 of the book.

Usage:
    python hyperparameter_tuning.py --config configs/hpo_config.yaml
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import itertools
from typing import Dict, List, Any


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


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


def train_and_evaluate(model, train_loader, val_loader, config, device):
    """Train model and return validation loss."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        
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
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 5):
                break
    
    return best_val_loss


def grid_search(param_grid: Dict[str, List[Any]], train_loader, val_loader, 
                horizon: int, device, base_config: dict):
    """Perform grid search over hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    results = []
    total_combinations = np.prod([len(v) for v in values])
    
    print(f"Running grid search over {total_combinations} configurations...")
    
    for i, combination in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combination))
        
        model = LSTMForecaster(
            input_size=1,
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            output_size=horizon,
            dropout=params.get('dropout', 0.2)
        ).to(device)
        
        config = base_config.copy()
        config.update(params)
        
        val_loss = train_and_evaluate(model, train_loader, val_loader, config, device)
        
        results.append({
            'params': params,
            'val_loss': val_loss
        })
        
        print(f"  [{i+1}/{total_combinations}] {params} -> Val Loss: {val_loss:.6f}")
    
    # Sort by validation loss
    results.sort(key=lambda x: x['val_loss'])
    return results


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    
    # Generate synthetic data
    n_samples = config.get('n_samples', 4380)  # 6 months for faster tuning
    time = np.arange(n_samples)
    daily_pattern = np.sin(2 * np.pi * time / 24)
    weekly_pattern = 0.5 * np.sin(2 * np.pi * time / (24 * 7))
    noise = 0.1 * np.random.randn(n_samples)
    load = 100 + 20 * daily_pattern + 10 * weekly_pattern + noise
    
    mean, std = load.mean(), load.std()
    load_normalized = (load - mean) / std
    
    lookback = config.get('lookback', 168)
    horizon = config.get('horizon', 24)
    X, y = create_sequences(load_normalized, lookback, horizon)
    
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train).unsqueeze(-1), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val).unsqueeze(-1), torch.FloatTensor(y_val))
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define hyperparameter search space
    param_grid = config.get('param_grid', {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.001, 0.0005],
        'dropout': [0.1, 0.2, 0.3]
    })
    
    print("\n=== Hyperparameter Tuning ===")
    print(f"Search space: {param_grid}")
    
    results = grid_search(param_grid, train_loader, val_loader, horizon, device, config)
    
    print("\n=== Top 5 Configurations ===")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result['params']} -> Val Loss: {result['val_loss']:.6f}")
    
    best_params = results[0]['params']
    print(f"\n=== Best Configuration ===")
    print(f"  Parameters: {best_params}")
    print(f"  Validation Loss: {results[0]['val_loss']:.6f}")
    
    # Calculate improvement
    baseline_loss = results[-1]['val_loss']
    best_loss = results[0]['val_loss']
    improvement = (baseline_loss - best_loss) / baseline_loss * 100
    print(f"\n  Improvement over worst: {improvement:.1f}%")
    print(f"\nExpected: 5-10% improvement through systematic optimization")
    
    return {'best_params': best_params, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--config', type=str, default='configs/hpo_config.yaml')
    args = parser.parse_args()
    main(args.config)
