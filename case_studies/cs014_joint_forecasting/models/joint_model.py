import torch
import torch.nn as nn

class JointForecaster(nn.Module):
    """
    Multi-stream forecaster for joint source, load, and price prediction.
    Uses a shared encoder to capture common temporal dependencies.
    """
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3):
        super(JointForecaster, self).__init__()
        
        # Shared Encoder (LSTM)
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Domain-Specific Heads
        self.load_head = nn.Linear(hidden_size, 1)
        self.solar_head = nn.Linear(hidden_size, 1)
        self.price_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.encoder(x)
        
        # Use top layer hidden state
        shared_features = hidden[-1]
        
        # Domain-specific outputs
        load_out = self.load_head(shared_features)
        solar_out = self.solar_head(shared_features)
        price_out = self.price_head(shared_features)
        
        # Concatenate for loss calculation
        return torch.cat([load_out, solar_out, price_out], dim=1) # (batch, 3)
