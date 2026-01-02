# CS011: Transformer-Based Renewable Energy Forecasting

This tutorial demonstrates how to apply the standard Transformer architecture to solar power forecasting. We explore how self-attention mechanisms can capture the complex periodicity and daily cycles inherent in renewable energy data.

## Learning Objectives
1. Adapt the vanilla Transformer for univariate renewable energy time series.
2. Understand the impact of positional encoding on solar energy patterns.
3. Handle the night-time sparsity (zeros) in solar forecasting.
4. Compare Transformer performance against baseline RNN/LSTM models.

## Structure
- `tutorial.ipynb`: Step-by-step Transformer implementation and training.
- `data_loader.py`: NREL solar data (D005) preparation.
- `model.py`: Transformer model wrapper and sequence-to-point training logic.
- `evaluation.py`: Performance assessment with a focus on WAPE and RMSE.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
