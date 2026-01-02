import torch
import numpy as np
import time

class ForecastingService:
    """
    Simulated forecasting service wrapper.
    In production, this would be a Flask, FastAPI, or TorchServe wrapper.
    """
    def __init__(self, model_path, scaler):
        self.model = self._load_model(model_path)
        self.scaler = scaler
        self.model.eval()

    def _load_model(self, model_path):
        # In a real scenario, we load the scripted or state_dict model
        # For this tutorial, we assume a compatible architecture
        from src.models.rnn import LSTMModel
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
        # model.load_state_dict(torch.load(model_path))
        return model

    def predict(self, historical_data):
        """
        Input: List or array of 168 historical load values.
        Output: Prediction for the next time step.
        """
        start_time = time.time()
        
        # Preprocessing
        x = np.array(historical_data).reshape(-1, 1)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output_scaled = self.model(x_tensor)
            
        # Postprocessing
        prediction = self.scaler.inverse_transform(output_scaled.numpy().reshape(-1, 1))
        
        latency = time.time() - start_time
        return float(prediction[0, 0]), latency

if __name__ == "__main__":
    # Mock usage
    print("Initializing Forecasting Service...")
    # service = ForecastingService("models/load_model.pt", scaler)
    # result, latency = service.predict([1000] * 168)
    # print(f"Prediction: {result}, Latency: {latency:.4f}s")
