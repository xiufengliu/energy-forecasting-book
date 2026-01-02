import numpy as np
import itertools
from src.models.rnn import LSTMModel
from src.utils.metrics import calculate_accuracy_metrics
import torch
import torch.nn as nn

def grid_search(param_grid, train_loader, val_loader, scaler, device='cpu'):
    """
    Perform a simple grid search for hyperparameters.
    """
    keys, values = zip(*param_grid.items())
    best_loss = float('inf')
    best_params = None
    
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"Testing params: {params}")
        
        # Instantiate model with current params
        model = LSTMModel(input_size=1, 
                          hidden_size=params['hidden_size'], 
                          num_layers=params['num_layers'], 
                          output_size=1).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        train_model(model, train_loader, optimizer, criterion, epochs=2, device=device)
        
        # Validate
        val_loss = validate_model(model, val_loader, criterion, device=device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            
    return best_params, best_loss

def train_model(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
    return total_loss / len(dataloader)
