import torch
import numpy as np

def evaluate_profit(model, dataloader, scaler, device='cpu'):
    """
    Simulate a simple battery storage arbitrage policy to calculate profit.
    Policy: Charge if forecast < threshold, Discharge if forecast > threshold.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y[:, 0, :].cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Rescale back to original prices ($/MWh)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Simple arbitrage threshold (e.g. mean price)
    threshold = np.mean(actuals_orig)
    
    def calculate_basket_profit(p_vec, a_vec):
        profit = 0
        for p, a in zip(p_vec, a_vec):
            if p > threshold: # Forecast says price is high -> Sell/Discharge
                profit += a
            elif p < threshold: # Forecast says price is low -> Buy/Charge
                profit -= a
        return profit

    model_profit = calculate_basket_profit(preds_orig, actuals_orig)
    
    # Benchmark: Perfect foresight (OR a simple MAE-optimized model)
    # Here we use actuals as 'forecast' for a theoretical maximum
    max_profit = calculate_basket_profit(actuals_orig, actuals_orig)
    
    return model_profit, max_profit * 0.7 # Simulate benchmark at 70% of theoretical max
