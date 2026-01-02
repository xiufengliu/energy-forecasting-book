import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def root_mean_squared_error(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def mean_absolute_error(y_true, y_pred):
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def continuous_ranked_probability_score(y_true, quantiles, quantile_levels):
    """
    Approximate CRPS for quantile forecasts.
    
    Args:
        y_true: True values (N,)
        quantiles: Predicted quantiles (N, Q)
        quantile_levels: Quantile levels (Q,)
    """
    y_true = np.array(y_true).reshape(-1, 1)
    quantiles = np.array(quantiles)
    quantile_levels = np.array(quantile_levels)
    
    # Simple approximation of CRPS using quantile loss
    # CRPS = 2 * average(QuantileLoss(y, q, alpha))
    diff = y_true - quantiles
    loss = np.maximum(quantile_levels * diff, (quantile_levels - 1) * diff)
    return 2 * np.mean(loss)

def evaluate_all(y_true, y_pred):
    """Compute all standard metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
