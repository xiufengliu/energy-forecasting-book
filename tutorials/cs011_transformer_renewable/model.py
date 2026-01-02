import torch
import torch.nn as nn
from src.models.transformer import TransformerModel

def build_renewable_transformer(input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1, device='cpu'):
    """
    Build the Transformer model for renewable forecasting.
    """
    model = TransformerModel(input_size, d_model, nhead, num_layers, dim_feedforward=128, output_size=output_size).to(device)
    return model

def train_transformer_step(model, x, y, optimizer, criterion):
    """
    Perform a single training step for Transformer.
    PyTorch Transformer expects (seq_len, batch, input_size).
    """
    model.train()
    optimizer.zero_grad()
    
    # Transpose input for standard Transformer: (batch, seq_len, input_size) -> (seq_len, batch, input_size)
    x = x.transpose(0, 1)
    
    outputs = model(x) # Output is (batch, output_size)
    
    # Simple mapping: predict last value or multi-step horizon
    # For this tutorial, we focus on the first predicted step
    loss = criterion(outputs, y[:, 0, :])
    loss.backward()
    optimizer.step()
    
    return loss.item()
