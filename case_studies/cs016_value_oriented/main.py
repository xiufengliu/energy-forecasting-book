import argparse
import torch
import torch.nn as nn
from data.data_loader import get_price_value_dataloader
from models.loss import ProfitAwareLoss
from evaluation.profit_evaluator import evaluate_profit

def main(args):
    """
    Main orchestration for Value-Oriented Forecasting.
    """
    # 1. Load Price Data
    train_loader, val_loader, test_loader, scaler = get_price_value_dataloader(
        args.batch_size, args.input_len, args.output_len
    )
    
    # 2. Initialize Model (Standard LSTM or Transformer)
    from src.models.rnn import LSTMModel
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(args.device)
    
    # 3. Profit-Aware Loss Function (Value-Oriented)
    # This loss penalizes errors that lead to bad business decisions (e.g. charging when price is high)
    criterion = ProfitAwareLoss(alpha=args.alpha)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training (Simulated)
    print(f"Training with Profit-Aware Loss (alpha={args.alpha})...")
    
    # 5. Evaluation (Profit-based metrics)
    profit, benchmark_profit = evaluate_profit(model, test_loader, scaler, args.device)
    print(f"Total Portfolio Profit: ${profit:.2f}")
    print(f"Benchmark (MAE-optimized) Profit: ${benchmark_profit:.2f}")
    print(f"Value Lift: {100 * (profit - benchmark_profit) / benchmark_profit:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value-Oriented Price Forecasting")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for profit-aware term")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_len", type=int, default=168)
    parser.add_argument("--output_len", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    main(args)
