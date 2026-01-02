import numpy as np
import matplotlib.pyplot as plt

def evaluate_price_tft(model, dataloader, scaler, device='cpu'):
    """
    Evaluate TFT for price forecasting.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).unsqueeze(-1)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Rescale
    # Scaler was fit on [price, load, temp], price is first column
    dummy = np.zeros((preds.shape[0] * preds.shape[1], 3))
    dummy[:, 0] = preds.flatten()
    preds_orig = scaler.inverse_transform(dummy)[:, 0].reshape(preds.shape)
    
    dummy[:, 0] = actuals.flatten()
    actuals_orig = scaler.inverse_transform(dummy)[:, 0].reshape(actuals.shape)
    
    mae = np.mean(np.abs(actuals_orig[:, 0, 0] - preds_orig[:, 0]))
    rmse = np.sqrt(np.mean((actuals_orig[:, 0, 0] - preds_orig[:, 0])**2))
    
    # Spike detection accuracy (e.g., price > 60)
    spikes_actual = actuals_orig[:, 0, 0] > 60
    spikes_pred = preds_orig[:, 0] > 60
    spike_acc = np.mean(spikes_actual == spikes_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'Spike Acc': spike_acc}, actuals_orig, preds_orig

def plot_price_forecast(y_actual, y_pred, title="Electricity Price Forecast - TFT"):
    """
    Plot actual vs predicted price.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[:, 0, 0], label='Actual', alpha=0.7)
    plt.plot(y_pred[:, 0], label='Predicted', alpha=0.7, linestyle='--')
    plt.title(title)
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
