import torch
import os

def deploy_model(model, export_path):
    """
    Prepare and save the model for production serving.
    """
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
    model_path = os.path.join(export_path, "load_forecaster.pt")
    
    # Export using TorchScript for optimized serving
    model.eval()
    # Simple dummy input for tracing (batch=1, seq_len=168, features=1)
    example_input = torch.rand(1, 168, 1)
    
    try:
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(model_path)
        print(f"Model successfully deployed to {model_path}")
    except Exception as e:
        print(f"TorchScript tracing failed: {e}. Saving standard state_dict instead.")
        torch.save(model.state_dict(), model_path.replace(".pt", "_state.pt"))
