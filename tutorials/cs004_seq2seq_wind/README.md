# CS004: Wind Power Forecasting with Seq2Seq

This tutorial demonstrates how to use Sequence-to-Sequence (Seq2Seq) models with attention for wind power forecasting. We use simulated NREL wind power data to illustrate the concepts.

## Learning Objectives
1. Understand the Seq2Seq architecture (Encoder-Decoder) for time series.
2. Implement attention mechanisms to capture temporal dependencies.
3. Handle the intermittency of wind power generation.
4. Evaluate model performance using standard energy forecasting metrics.

## Structure
- `tutorial.ipynb`: Main tutorial notebook with step-by-step explanations.
- `data_preprocessing.py`: Data loading and preparation utilities.
- `model.py`: Seq2Seq model wrapper and training logic.
- `evaluation.py`: Performance assessment and visualization.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
