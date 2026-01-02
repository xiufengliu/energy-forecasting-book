import numpy as np
import matplotlib.pyplot as plt

def evaluate_transformer_renewable(model, dataloader, scaler, device='cpu'):
    """
    Evaluate Transformer results for renewable forecasting.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).transpose(0, 1) # (seq_len, batch, input_size)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Rescale
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
    
    mae = np.mean(np.abs(actuals_orig[:, 0, 0] - preds_orig[:, 0]))
    wape = 100 * np.sum(np.abs(actuals_orig[:, 0, 0] - preds_orig[:, 0])) / np.sum(actuals_orig[:, 0, 0])
    
    return {'MAE': mae, 'WAPE': wape}, actuals_orig, preds_orig

def plot_transformer_renewable(y_actual, y_pred, title="Solar Forecast - Transformer"):
    """
    Plot actual vs predicted.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[:, 0, 0], label='Actual', alpha=0.7)
    plt.plot(y_pred[:, 0], label='Predicted', alpha=0.7, linestyle='--')
    plt.title(title)
    plt.ylabel('Solar Power (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
