# CS006: Probabilistic Load Forecasting

This tutorial demonstrates how to quantify uncertainty in load forecasting using probabilistic deep learning. We implement quantile regression with an LSTM model to generate prediction intervals.

## Learning Objectives
1. Understand the importance of uncertainty quantification in energy systems.
2. Implement Pinball Loss (Quantile Loss) in PyTorch.
3. Build a multi-quantile forecasting model.
4. Evaluate probabilistic performance using coverage and prediction interval widths.

## Structure
- `tutorial.ipynb`: Main tutorial notebook with step-by-step explanations.
- `data_preprocessing.py`: PJM load data preparation.
- `model.py`: Quantile Regression model wrapper and Pinball Loss.
- `evaluation.py`: Coverage analysis and uncertainty visualization.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
