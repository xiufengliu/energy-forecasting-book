import torch
import torch.nn as nn
from src.models.rnn import LSTMModel

class QuantileLoss(nn.Module):
    """
    Pinball Loss for quantile regression.
    """
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds: (batch, output_len, num_quantiles)
        # target: (batch, output_len, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i:i+1]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

def build_probabilistic_model(input_size=1, hidden_size=64, num_layers=2, num_quantiles=3, device='cpu'):
    """
    Build an LSTM model that outputs multiple quantiles.
    """
    # Output size is num_quantiles
    model = LSTMModel(input_size, hidden_size, num_layers, num_quantiles).to(device)
    return model

def train_probabilistic_step(model, x, y, optimizer, criterion):
    """
    Perform a single training step for probabilistic forecasting.
    """
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x) # (batch, output_len, num_quantiles)
    
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()
