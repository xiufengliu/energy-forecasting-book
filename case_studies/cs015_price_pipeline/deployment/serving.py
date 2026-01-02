import torch
import os

def deploy_price_model(model, export_path):
    """
    Prepare price model for serving.
    """
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
    model_path = os.path.join(export_path, "price_forecaster.pt")
    
    # Export for production inference
    torch.save(model.state_dict(), model_path)
    print(f"Price forecasting model successfully saved to {model_path}")
