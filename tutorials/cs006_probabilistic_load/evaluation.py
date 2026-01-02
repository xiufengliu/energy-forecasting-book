import numpy as np
import matplotlib.pyplot as plt

def evaluate_probabilistic_model(model, x_test, y_test, scaler, quantiles=[0.1, 0.5, 0.9], device='cpu'):
    """
    Evaluate the probabilistic model and return quantiles and metrics.
    """
    model.eval()
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    
    with torch.no_grad():
        outputs = model(x_test_tensor)
    
    predictions = outputs.cpu().numpy()
    
    # Rescale back to original values
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    predictions_orig = []
    for i in range(len(quantiles)):
        q_preds = scaler.inverse_transform(predictions[:, :, i].reshape(-1, 1)).reshape(predictions.shape[:-1])
        predictions_orig.append(q_preds)
    
    predictions_orig = np.stack(predictions_orig, axis=-1)
    
    # Calculate coverage (for 80% interval: 0.1 to 0.9)
    # Simple coverage check for demonstration
    low_q, high_q = predictions_orig[:, 0, 0], predictions_orig[:, 0, 2]
    actual = y_test_orig[:, 0, 0]
    coverage = np.mean((actual >= low_q) & (actual <= high_q))
    
    return coverage, y_test_orig, predictions_orig

def plot_probabilistic_forecast(y_actual, predictions_orig, quantiles=[0.1, 0.5, 0.9], title='Probabilistic Load Forecast'):
    """
    Plot actual vs predicted median and prediction intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[:, 0, 0], label='Actual', color='black', alpha=0.8)
    
    plt.plot(predictions_orig[:, 0, 1], label='Median (q=0.5)', color='blue', alpha=0.7)
    
    # Fill between 10th and 90th quantiles
    plt.fill_between(
        range(len(y_actual)), 
        predictions_orig[:, 0, 0], 
        predictions_orig[:, 0, 2], 
        color='blue', alpha=0.2, label='80% Interval'
    )
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
