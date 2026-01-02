import numpy as np
import matplotlib.pyplot as plt

def evaluate_multivariate_faformer(model, dataloader, scaler, device='cpu'):
    """
    Evaluate FAformer for multivariate forecasting.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Rescale back to original values per zone
    preds_orig = scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, actuals.shape[-1])).reshape(actuals.shape)
    
    # Calculate MAE per zone
    zone_metrics = {}
    for i in range(actuals_orig.shape[-1]):
        mae = np.mean(np.abs(actuals_orig[:, 0, i] - preds_orig[:, i]))
        zone_metrics[f'zone{i+1}'] = {'MAE': mae}
        
    return zone_metrics, actuals_orig, preds_orig

def plot_faformer_results(y_actual, y_pred, zone_idx=0, title="Multivariate Forecast - FAformer"):
    """
    Plot actual vs predicted for a specific zone.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[:, 0, zone_idx], label='Actual', alpha=0.7)
    plt.plot(y_pred[:, zone_idx], label='Predicted', alpha=0.7, linestyle='--')
    plt.title(f"{title} - Zone {zone_idx+1}")
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
