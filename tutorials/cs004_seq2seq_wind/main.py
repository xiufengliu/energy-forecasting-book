"""
Seq2Seq Wind Power Forecasting - Main Entry Point
==================================================
This script provides a CLI interface for Seq2Seq-based wind power forecasting,
reproducing the results discussed in Chapter 4 of the book.

Usage:
    python main.py --config configs/seq2seq_wind.yaml
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * keys, dim=1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(output_size + hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        context, _ = self.attention(hidden[-1], encoder_outputs)
        lstm_input = torch.cat([x, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2SeqForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(output_size, hidden_size, num_layers, dropout)
        self.horizon = horizon
        self.output_size = output_size
    
    def forward(self, x, teacher_forcing_ratio: float = 0.0):
        batch_size = x.size(0)
        encoder_outputs, hidden, cell = self.encoder(x)
        
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        predictions = []
        
        for t in range(self.horizon):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            predictions.append(prediction)
            decoder_input = prediction.unsqueeze(1)
        
        return torch.stack(predictions, dim=1).squeeze(-1)


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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_seq2seq.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_seq2seq.pt'))
    return model


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(config.get('seed', 42))
    n_samples = config.get('n_samples', 8760)
    time = np.arange(n_samples)
    
    # Simulate wind power with variable patterns
    daily_pattern = 0.3 * np.sin(2 * np.pi * time / 24)
    wind_base = 30 + 20 * np.sin(2 * np.pi * time / (24 * 3))  # 3-day weather cycles
    noise = 10 * np.random.randn(n_samples)
    wind = np.clip(wind_base + daily_pattern + noise, 0, 100)
    
    mean, std = wind.mean(), wind.std()
    wind_normalized = (wind - mean) / std
    
    lookback = config.get('lookback', 168)
    horizon = config.get('horizon', 24)
    X, y = create_sequences(wind_normalized, lookback, horizon)
    
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
    
    model = Seq2SeqForecaster(
        input_size=1,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        output_size=1,
        horizon=horizon,
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
    
    wape = calculate_wape(actuals, predictions)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    print(f"\nTest Results:")
    print(f"  WAPE: {wape:.2f}%")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"\nExpected WAPE range: 12% - 18%")
    
    return {'wape': wape, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq Wind Power Forecasting')
    parser.add_argument('--config', type=str, default='configs/seq2seq_wind.yaml')
    args = parser.parse_args()
    main(args.config)
