# CS005: Multivariate Load Forecasting

This tutorial demonstrates how to handle multiple related time series (multivariate forecasting) using deep learning. We use simulated data inspired by the GEFCom2014 regions and incorporate weather features as external drivers.

## Learning Objectives
1. Understand multivariate time series structures in energy forecasting.
2. Implement Channel-Independent vs. Channel-Dependent forecasting strategies.
3. Integrate exogenous features (e.g., temperature) into forecasting models.
4. Evaluate multivariate performance across different regions/zones.

## Structure
- `tutorial.ipynb`: Main tutorial notebook with step-by-step explanations.
- `data_preprocessing.py`: Multi-zone data loading and feature engineering.
- `model.py`: Multivariate model wrappers (Channel-Independent LSTM).
- `evaluation.py`: Per-zone and aggregate performance assessment.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
