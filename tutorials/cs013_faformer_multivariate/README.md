# CS013: Multivariate Energy Forecasting with Frequency-Aware Transformer (FAformer)

This tutorial demonstrates how to use the Frequency-Aware Transformer (FAformer) for multivariate load forecasting. FAformer is designed to capture seasonal and periodic patterns by incorporating frequency-domain information directly into the attention mechanism.

## Learning Objectives
1. Implement the Frequency-Aware Attention mechanism using Fast Fourier Transforms (FFT).
2. Understand how to identify dominant periodicities in multivariate energy data.
3. Combine frequency-domain features with standard temporal self-attention.
4. Evaluate multivariate performance across zones with different seasonal signatures.

## Structure
- `tutorial.ipynb`: Walkthrough of frequency-aware forecasting.
- `data_loader.py`: GEFCom multivariate load data preparation (D003).
- `model.py`: FAformer model wrapper and training logic.
- `evaluation.py`: Accuracy metrics and frequency-domain analysis.

## Setup
Ensure you have the required dependencies installed (see `requirements.txt` in the root `code/` directory).

```bash
pip install -r requirements.txt
```

## Running the Tutorial
Open `tutorial.ipynb` in a Jupyter environment and follow the instructions.
