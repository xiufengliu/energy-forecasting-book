import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import calculate_accuracy_metrics

def evaluate_multivariate_model(model, x_test, y_test, scaler_y, device='cpu'):
    """
    Evaluate the multivariate model and return metrics and predictions.
    """
    model.eval()
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    
    with torch.no_grad():
        outputs = model(x_test_tensor)
    
    predictions = outputs.cpu().numpy()
    
    # Rescale back to original values for each zone
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
    predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    
    zone_metrics = {}
    for i in range(y_test.shape[-1]):
        zone_name = f'zone{i+1}'
        zone_metrics[zone_name] = calculate_accuracy_metrics(y_test_orig[:, 0, i], predictions_orig[:, 0, i])
        
    total_metrics = calculate_accuracy_metrics(y_test_orig.flatten(), predictions_orig.flatten())
    
    return zone_metrics, total_metrics, y_test_orig, predictions_orig

def plot_multivariate_forecast(y_actual, y_pred, zone_idx=0, title='Multivariate Load Forecast'):
    """
    Plot actual vs predicted values for a specific zone.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[:, 0, zone_idx], label='Actual', alpha=0.7)
    plt.plot(y_pred[:, 0, zone_idx], label='Predicted', alpha=0.7, linestyle='--')
    plt.title(f'{title} - Zone {zone_idx+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
