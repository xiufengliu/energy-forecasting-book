import torch
import numpy as np
from src.utils.metrics import calculate_accuracy_metrics

def evaluate_price_pipeline(model, dataloader, scaler):
    """
    Evaluate price forecasting model with a focus on volatility.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y[:, 0, :].cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Rescale
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
    
    metrics = calculate_accuracy_metrics(actuals_orig, preds_orig)
    
    # Custom price metric: Mean Absolute Error on Spikes (actual > 100)
    spike_mask = actuals_orig > 100
    if np.any(spike_mask):
        spike_mae = np.mean(np.abs(actuals_orig[spike_mask] - preds_orig[spike_mask]))
        metrics['Spike_MAE'] = spike_mae
    else:
        metrics['Spike_MAE'] = 0.0
        
    return metrics
