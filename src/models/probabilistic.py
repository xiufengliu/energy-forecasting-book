import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert preds.shape[0] == target.shape[0]
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.cat(losses, dim=1).sum(dim=1))
        return loss

class ProbabilisticLSTM(nn.Module):
    """LSTM model predicting multiple quantiles."""
    def __init__(self, input_size, hidden_size, num_layers, num_quantiles, dropout=0.2):
        super(ProbabilisticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_quantiles)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
