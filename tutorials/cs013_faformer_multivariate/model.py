import torch
import torch.nn as nn
from src.models.faformer import FAformerModel

def build_multivariate_faformer(input_size=3, d_model=64, nhead=4, num_layers=2, output_size=3, top_k=5, device='cpu'):
    """
    Build the FAformer model for multivariate forecasting.
    """
    model = FAformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=128,
        output_size=output_size,
        top_k=top_k
    ).to(device)
    return model

def train_faformer_step(model, x, y, optimizer, criterion):
    """
    Perform a single training step for FAformer.
    x: (batch, seq_len, input_size)
    y: (batch, output_len, output_size)
    """
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x) # Output is (batch, output_size)
    
    # Map last step of y to prediction
    loss = criterion(outputs, y[:, 0, :])
    loss.backward()
    optimizer.step()
    
    return loss.item()
