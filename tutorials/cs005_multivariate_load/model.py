import torch
import torch.nn as nn
from src.models.rnn import LSTMModel
from src.models.multivariate import ChannelIndependentWrapper

def build_multivariate_model(input_size=1, hidden_size=64, num_layers=2, num_channels=4, output_size=1, device='cpu'):
    """
    Build a multivariate model using channel-independent LSTM.
    Input size is 1 because each channel is processed independently.
    """
    base_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = ChannelIndependentWrapper(base_model, num_channels).to(device)
    return model

def train_multivariate_step(model, x, y, optimizer, criterion):
    """
    Perform a single training step for multivariate data.
    """
    model.train()
    optimizer.zero_grad()
    
    # x: (batch, seq_len, num_channels)
    # y: (batch, output_len, num_channels) - here we assume mapping to last step
    
    outputs = model(x) # (batch, output_dim, num_channels)
    
    # Simple loss calculation - mapping last prediction to target
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()
