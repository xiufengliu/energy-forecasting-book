# CS016: Value-Oriented Electricity Price Forecasting

This advanced case study demonstrates the concept of **Value-Oriented Forecasting**. Unlike traditional models that minimize statistical errors (MAE, RMSE), this project optimizes model parameters specifically to maximize downstream monetary profit in a battery storage arbitrage scenario.

## Key Features
- **Profit-Aware Loss Function**: Implements a custom PyTorch loss function that penalizes forecasting errors based on their financial consequences.
- **Decision-Driven Modeling**: Bridges the gap between forecasting and decision-making by incorporating battery storage constraints into the training loop.
- **Financial Evaluation**: Measures performance in terms of real-world profit ($) and "Value Lift" compared to accuracy-optimized benchmarks.
- **Arbitrage Simulation**: Includes a simplified energy storage policy simulator to validate the monetary impact of the forecasts.

## Project Structure
- `main.py`: Orchestrates the value-oriented training and financial evaluation.
- `data/data_loader.py`: Volatile price data preparation (D004).
- `models/loss.py`: Custom Profit-Aware Loss implementation.
- `evaluation/profit_evaluator.py`: Battery storage arbitrage simulator and profit calculator.

## Usage
Run the value-oriented forecasting simulation:
```bash
python main.py --alpha 0.7 --epochs 10
```

## Significance
In the energy sector, a 1% improvement in MAE does not always translate to a 1% increase in profit. Value-oriented forecasting ensures that deep learning models are aligned with the ultimate business goals of the organization, particularly in highly volatile markets.
