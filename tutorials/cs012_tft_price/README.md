# CS012: Electricity Price Forecasting with Temporal Fusion Transformer (TFT)

This tutorial demonstrates how to use the Temporal Fusion Transformer (TFT) for the challenging task of electricity price forecasting. TFT is particularly suited for this because it can effectively handle multiple time-varying features and static metadata.

## Learning Objectives
1. Implement the Temporal Fusion Transformer (TFT) architecture.
2. Understand Variable Selection Networks (VSN) for identifying key price drivers.
3. Handle price spikes and high volatility through robust modeling.
4. Interpret model focus through multiple attention heads.

## Structure
- `tutorial.ipynb`: Deep dive into TFT price forecasting.
- `data_loader.py`: Multivariate price data preparation (D004).
- `model.py`: TFT model wrapper and training orchestration.
- `evaluation.py`: Accuracy metrics with price spike analysis.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
