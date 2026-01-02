# CS007: End-to-End Load Forecasting Pipeline

This case study implements a complete, production-ready load forecasting pipeline. It demonstrates the transition from model training to deployment.

## Key Features
- **Modular Architecture**: Separate modules for data loading, model selection, evaluation, and serving.
- **Support for Multiple Models**: Easily switch between LSTM and Transformer architectures.
- **Production-Ready Serving**: Model export using TorchScript for optimized deployment.
- **Evaluation Framework**: Comprehensive accuracy metrics calculation on test data.

## Project Structure
- `main.py`: Pipeline orchestration and CLI entry point.
- `data/data_loader.py`: Handles dataset-specific loading and scaling (D002 - ERCOT).
- `models/selector.py`: Factory for instantiating different architectures.
- `evaluation/evaluator.py`: Performance assessment module.
- `deployment/serving.py`: Model export and deployment utilities.

## Usage
Run the pipeline with default settings:
```bash
python main.py --model_type transformer --dataset ERCOT
```

## Deployment
The model is exported to `models/export/load_forecaster.pt` by default, ready for use in a production serving environment.
