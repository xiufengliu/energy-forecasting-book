# CS015: Specialized Electricity Price Forecasting Pipeline

This case study implements a production-grade forecasting pipeline specifically optimized for electricity market prices. It addresses the unique challenges of price forecasting, such as high volatility, sudden spikes, and strong periodicities across different time scales.

## Key Features
- **Price-Optimized Architectures**: Supports FAformer and Transformer models, which excel at capturing market dynamics and seasonal shifts.
- **Volatility-Aware Evaluation**: Implements specialized metrics like Spike-MAE to assess model performance during high-price events.
- **Robust Preprocessing**: Handles electricity price data characterized by non-linear trends and extreme outliers.
- **Market-Ready Serving**: Streamlined model export for low-latency market participation.

## Project Structure
- `main.py`: Pipeline entry point and orchestration.
- `data/data_loader.py`: Specialized NordPool/GEFCom price data (D004) preparation.
- `models/selector.py`: Factory for price-specific deep learning architectures.
- `evaluation/evaluator.py`: Performance module with volatility and spike analysis.
- `deployment/serving.py`: Market-ready deployment utilities.

## Usage
Run price forecasting for the NordPool market:
```bash
python main.py --market NordPool --model_type faformer
```

## Significance
Electricity prices are among the most volatile commodities. A specialized pipeline that can accurately predict both base prices and extreme spikes is critical for market participants, utilities, and large-scale industrial consumers.
