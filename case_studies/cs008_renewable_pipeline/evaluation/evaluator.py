import torch
import numpy as np

def evaluate_renewable_pipeline(model, dataloader, scaler):
    """
    Evaluate renewable forecasting model with robust metrics.
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
    
    # Rescale back to original values
    # Scaler was fit on [y, nwp_feature], so we use first column for y
    dummy = np.zeros((preds.shape[0] * preds.shape[1], 2))
    dummy[:, 0] = preds.flatten()
    preds_orig = scaler.inverse_transform(dummy)[:, 0].reshape(preds.shape)
    
    dummy[:, 0] = actuals.flatten()
    actuals_orig = scaler.inverse_transform(dummy)[:, 0].reshape(actuals.shape)
    
    # Calculate robust metrics
    mae = np.mean(np.abs(actuals_orig - preds_orig))
    rmse = np.sqrt(np.mean((actuals_orig - preds_orig)**2))
    
    # WAPE (Weighted Absolute Percentage Error)
    wape = 100 * np.sum(np.abs(actuals_orig - preds_orig)) / np.sum(actuals_orig)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'WAPE': wape
    }
