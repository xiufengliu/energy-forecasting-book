import torch
import torch.nn as nn

class ProfitAwareLoss(nn.Module):
    """
    Value-Oriented Loss function for price forecasting.
    Combines MSE with a penalty for 'decision-flipping' errors.
    """
    def __init__(self, alpha=0.5):
        super(ProfitAwareLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # preds: (batch, 1) rescaled or raw?
        # Let's assume raw or similarly scaled
        
        # 1. Standard Statistical Loss
        mse_loss = self.mse(preds, targets)
        
        # 2. Decision-Based Penalty (Value-Oriented)
        # Simple heuristic: penalize when forecast and target are on opposite sides of the mean price
        # (Assuming the mean price is the decision threshold for a simple arbitrage policy)
        
        # We use a soft differentiable approximation of the sign flip
        decision_penalty = torch.mean(torch.relu(-preds * targets)) # Penalize opposite signs if centered at 0
        
        # Total loss
        return (1 - self.alpha) * mse_loss + self.alpha * decision_penalty
