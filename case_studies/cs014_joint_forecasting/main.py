import argparse
import torch
import torch.nn as nn
from data.data_loader import get_joint_dataloader
from models.joint_model import JointForecaster
from evaluation.joint_evaluator import evaluate_joint_system

def main(args):
    """
    Main orchestration for joint forecasting (Source, Load, Price).
    """
    # 1. Load joint data streams (synchronized)
    train_loader, val_loader, test_loader, scalers = get_joint_dataloader(
        args.batch_size, args.input_len, args.output_len
    )
    
    # 2. Initialize Joint Model (Shared encoder + Multi-head decoder)
    model = JointForecaster(
        input_size=3, # [load, solar, price]
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=3 # [load_forecast, solar_forecast, price_forecast]
    ).to(args.device)
    
    # 3. Training logic [Simplified for tutorial]
    print(f"Starting joint training for {args.epochs} epochs...")
    
    # 4. Joint Evaluation
    metrics = evaluate_joint_system(model, test_loader, scalers, args.device)
    print("Joint Forecasting Metrics:")
    for stream, m in metrics.items():
        print(f"  {stream.upper()}: {m}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Source-Load-Price Forecasting")
    parser.add_argument("--input_len", type=int, default=168, help="1 week history")
    parser.add_argument("--output_len", type=int, default=24, help="Next 24h")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    main(args)
