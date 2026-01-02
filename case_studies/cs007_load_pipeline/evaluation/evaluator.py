import torch
import numpy as np
from src.utils.metrics import calculate_accuracy_metrics

def evaluate_pipeline(model, dataloader, scaler):
    """
    Evaluate the model on a dataloader.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Inverse transform
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    
    # Calculate metrics for the first time step of the horizon
    metrics = calculate_accuracy_metrics(actuals_orig[:, 0, 0], preds_orig[:, 0, 0])
    return metrics
