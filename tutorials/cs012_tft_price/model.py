import torch
import torch.nn as nn
from src.models.tft import TemporalFusionTransformer

def build_price_tft(num_vars=3, input_size=1, d_model=64, n_head=4, output_size=1, device='cpu'):
    """
    Build the Temporal Fusion Transformer for price forecasting.
    """
    config = {
        'num_vars': num_vars,
        'input_size': input_size,
        'd_model': d_model,
        'n_head': n_head,
        'output_size': output_size
    }
    model = TemporalFusionTransformer(config).to(device)
    return model

def train_tft_step(model, x, y, optimizer, criterion):
    """
    Perform a single training step for TFT.
    x: (batch, seq_len, num_vars)
    y: (batch, output_len, 1)
    """
    model.train()
    optimizer.zero_grad()
    
    # Reshape x to (batch, seq_len, num_vars, input_size) as expected by our TFT implementation
    x = x.unsqueeze(-1)
    
    outputs = model(x) # Output is (batch, output_size)
    
    # Map last step of y to prediction
    loss = criterion(outputs, y[:, 0, :])
    loss.backward()
    optimizer.step()
    
    return loss.item()
