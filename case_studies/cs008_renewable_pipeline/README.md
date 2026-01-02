# CS008: End-to-End Renewable Forecasting Pipeline

This case study implements a production-ready renewable energy forecasting pipeline for solar and wind power. It emphasizes the integration of weather forecasts and robust performance evaluation.

## Key Features
- **NWP Integration**: Demonstrates how to incorporate Numerical Weather Prediction (irradiance, wind speed) as exogenous features.
- **Dual Support**: Handles both solar and wind power forecasting patterns.
- **Robust Evaluation**: Uses Weighted Absolute Percentage Error (WAPE) and WMAE to handle the intermittency (zeros) of renewable generation.
- **Automated Export**: Saves trained models in a format suitable for production serving.

## Project Structure
- `main.py`: Orchestrates the renewable forecasting workflow.
- `data/data_loader.py`: Loads solar (D005) or wind (D006) data with NWP features.
- `models/selector.py`: Instantiates architectures suitable for periodic/stochastic series (TCN, Transformer).
- `evaluation/evaluator.py`: Implements metrics robust to zero-values in raw data.
- `deployment/serving.py`: Model export utilities.

## Usage
Run solar forecasting:
```bash
python main.py --source_type solar --model_type tcn
```

Run wind forecasting:
```bash
python main.py --source_type wind --model_type transformer
```

## Robust Metrics
For renewable forecasting, traditional MAPE often fails due to zero values at night or during still periods. This pipeline primarily reports **WAPE**, which provides a stable percentage error by normalizing total absolute error by total actual generation.
