import argparse
import torch
from data.data_loader import get_renewable_dataloader
from models.selector import get_model
from evaluation.evaluator import evaluate_renewable_pipeline
from deployment.serving import deploy_renewable_model

def main(args):
    """
    Main entry point for the renewable energy forecasting pipeline.
    """
    # 1. Data Preparation (Solar/Wind + NWP)
    train_loader, val_loader, test_loader, scaler = get_renewable_dataloader(
        args.source_type, args.input_len, args.output_len
    )
    
    # 2. Model Selection (optimized for intermittency)
    model = get_model(args.model_type, args.input_size, args.hidden_size, args.num_layers)
    
    # 3. Training/Loading
    print(f"Executing {args.source_type} forecasting with {args.model_type}...")
    
    # 4. Evaluation (using WAPE/WMAE for robustness)
    metrics = evaluate_renewable_pipeline(model, test_loader, scaler)
    print(f"Renewable Pipeline Performance: {metrics}")
    
    # 5. Deployment
    deploy_renewable_model(model, args.export_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Renewable Forecasting Pipeline")
    parser.add_argument("--source_type", type=str, default="solar", choices=["solar", "wind"], help="Energy source")
    parser.add_argument("--model_type", type=str, default="tcn", help="Model architecture")
    parser.add_argument("--input_len", type=int, default=48, help="Historical window length")
    parser.add_argument("--output_len", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--export_path", type=str, default="models/export/", help="Path to export model")
    
    args = parser.parse_args()
    main(args)
