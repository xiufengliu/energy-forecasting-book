import argparse
import torch
from data.data_loader import get_price_pipeline_dataloader
from models.selector import get_model
from evaluation.evaluator import evaluate_price_pipeline
from deployment.serving import deploy_price_model

def main(args):
    """
    Main entry point for the specialized price forecasting pipeline.
    """
    # 1. Data Preparation (Price-specific features)
    train_loader, val_loader, test_loader, scaler = get_price_pipeline_dataloader(
        args.market, args.input_len, args.output_len
    )
    
    # 2. Model Selection (e.g., TFT or FAformer)
    model = get_model(args.model_type, args.input_size, args.hidden_size, args.num_layers)
    
    # 3. Training/Loading
    print(f"Training {args.model_type} for {args.market} market...")
    
    # 4. Evaluation (MAE, RMSE, Spike Accuracy)
    metrics = evaluate_price_pipeline(model, test_loader, scaler)
    print(f"Price Pipeline Metrics: {metrics}")
    
    # 5. Deployment
    deploy_price_model(model, args.export_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Electricity Price Forecasting Pipeline")
    parser.add_argument("--market", type=str, default="NordPool", help="Market identifier (D004)")
    parser.add_argument("--model_type", type=str, default="faformer", help="Model architecture")
    parser.add_argument("--input_len", type=int, default=168)
    parser.add_argument("--output_len", type=int, default=24)
    parser.add_argument("--export_path", type=str, default="models/export/")
    
    args = parser.parse_args()
    main(args)
