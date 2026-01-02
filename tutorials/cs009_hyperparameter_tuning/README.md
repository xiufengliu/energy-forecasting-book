# CS009: Hyperparameter Tuning for Energy Forecasting

This tutorial demonstrates systematic strategies for optimizing the hyperparameters of deep learning models in energy forecasting. We implement a grid search approach to find the best configuration for an LSTM-based load forecaster.

## Learning Objectives
1. Understand the impact of hyperparameters (learning rate, hidden size, layers) on model performance.
2. Implement a basic grid search framework for PyTorch models.
3. Use temporal validation sets to prevent data leakage during tuning.
4. Visualize and interpret tuning results to select a final model.

## Structure
- `tutorial.ipynb`: Step-by-step optimization walkthrough.
- `tuning.py`: Grid search and training loop logic.
- `data_loader.py`: PJM dataset preparation for tuning.
- `evaluation.py`: Results summarization and visualization.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
