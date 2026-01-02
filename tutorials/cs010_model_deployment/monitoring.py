import numpy as np
from collections import deque

class ModelMonitor:
    """
    Performance monitoring for production energy forecasting models.
    Tracks latency, absolute error, and robust MAPE.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)

    def record_metrics(self, prediction, actual, latency):
        """
        Record a single inference result and its corresponding ground truth.
        """
        self.latencies.append(latency)
        self.predictions.append(prediction)
        
        if actual is not None:
            self.actuals.append(actual)
            self.errors.append(abs(prediction - actual))

    def get_summary(self, epsilon=1e-6):
        """
        Return a summary of model performance within the current window.
        """
        if not self.errors:
            return {"status": "No feedback data yet"}
            
        avg_latency = np.mean(self.latencies)
        p95_latency = np.percentile(self.latencies, 95)
        mae = np.mean(self.errors)
        
        # Robust MAPE handler (handles zeros in actuals using epsilon)
        mape = np.mean(np.array(self.errors) / (np.abs(self.actuals) + epsilon)) * 100
        
        return {
            "avg_latency": f"{avg_latency:.4f}s",
            "p95_latency": f"{p95_latency:.4f}s",
            "MAE": f"{mae:.2f} MW",
            "MAPE": f"{mape:.2f}%",
            "window_count": len(self.errors)
        }

    def check_alerts(self, thresholds):
        """
        Check if any metrics exceed predefined thresholds.
        """
        summary = self.get_summary()
        if "status" in summary: return []
        
        alerts = []
        if float(summary["avg_latency"][:-1]) > thresholds["latency"]:
            alerts.append("CRITICAL: High latency detected")
        if float(summary["MAPE"][:-1]) > thresholds["mape"]:
            alerts.append("WARNING: Model drift detected (High MAPE)")
            
        return alerts
