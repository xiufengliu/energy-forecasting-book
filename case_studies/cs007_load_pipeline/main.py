import argparse
import torch
import torch.nn as nn
from data.data_loader import get_dataloader
from models.selector import get_model
from evaluation.evaluator import evaluate_pipeline
from deployment.serving import deploy_model

def main(args):
    """
    Main entry point for the load forecasting pipeline.
    """
    # 1. Data Preparation
    train_loader, val_loader, test_loader, scaler = get_dataloader(args.dataset, args.input_len, args.output_len)
    
    # 2. Model Selection
    model = get_model(args.model_type, args.input_size, args.hidden_size, args.num_layers)
    
    # 3. Training/Loading
    # [Simplified: assume model is already trained or training logic is here]
    print(f"Training/Loading {args.model_type} model...")
    
    # 4. Evaluation
    metrics = evaluate_pipeline(model, test_loader, scaler)
    print(f"Pipeline Performance: {metrics}")
    
    # 5. Deployment
    deploy_model(model, args.export_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Load Forecasting Pipeline")
    parser.add_argument("--dataset", type=str, default="ERCOT", help="Dataset identifier (D002)")
    parser.add_argument("--model_type", type=str, default="transformer", help="Model architecture")
    parser.add_argument("--input_len", type=int, default=168, help="Historical window length")
    parser.add_argument("--output_len", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--export_path", type=str, default="models/export/", help="Path to export model")
    
    args = parser.parse_args()
    main(args)
