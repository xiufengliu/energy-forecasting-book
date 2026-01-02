import torch
import numpy as np
from src.utils.metrics import calculate_accuracy_metrics

def evaluate_joint_system(model, dataloader, scalers, device='cpu'):
    """
    Evaluate the joint forecasting model across all three streams.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y[:, 0, :].cpu().numpy()) # Compare last step
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Calculate stream-specific metrics
    metrics = {}
    
    # Load
    load_preds = scalers['load'].inverse_transform(preds[:, 0:1])
    load_actuals = scalers['load'].inverse_transform(actuals[:, 0:1])
    metrics['load'] = calculate_accuracy_metrics(load_actuals, load_preds)
    
    # Solar
    solar_preds = scalers['solar'].inverse_transform(preds[:, 1:2])
    solar_actuals = scalers['solar'].inverse_transform(actuals[:, 1:2])
    metrics['solar'] = calculate_accuracy_metrics(solar_actuals, solar_preds)
    
    # Price
    price_preds = scalers['price'].inverse_transform(preds[:, 2:3])
    price_actuals = scalers['price'].inverse_transform(actuals[:, 2:3])
    metrics['price'] = calculate_accuracy_metrics(price_actuals, price_preds)
    
    return metrics
