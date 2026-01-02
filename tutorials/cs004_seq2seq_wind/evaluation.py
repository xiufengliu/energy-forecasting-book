import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import calculate_accuracy_metrics

def evaluate_model(model, x_test, y_test, scaler, device='cpu'):
    """
    Evaluate the model on test data and return metrics and predictions.
    """
    model.eval()
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    
    with torch.no_grad():
        outputs = model(x_test_tensor, y_test.shape[1], teacher_forcing_ratio=0)
    
    predictions = outputs.cpu().numpy()
    
    # Rescale back to original values
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    
    metrics = calculate_accuracy_metrics(y_test_orig[:, 0, :], predictions_orig[:, 0, :])
    
    return metrics, y_test_orig, predictions_orig

def plot_forecast(y_actual, y_pred, title='Wind Power Forecast'):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7, linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Wind Power (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
