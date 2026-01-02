import torch
import os

def deploy_renewable_model(model, export_path):
    """
    Prepare renewable model for serving.
    """
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
    model_path = os.path.join(export_path, "renewable_forecaster.pt")
    
    # Simple export for demonstration
    torch.save(model.state_dict(), model_path)
    print(f"Renewable model saved to {model_path}")
