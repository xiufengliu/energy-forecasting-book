# CS014: Joint Source-Load-Price Forecasting

This case study implements a sophisticated joint forecasting system that simultaneously predicts energy demand (load), renewable generation (solar), and electricity market prices. This holistic approach captures the interdependencies between these critical energy system variables.

## Key Features
- **Data Synchronization**: Aligns disparate data streams (PJM Load, NREL Solar, GEFCom Price) into a unified temporal training set.
- **Shared Representation**: Uses a shared LSTM encoder to learn common temporal features relevant to all energy streams.
- **Multi-Head Architecture**: Implements domain-specific prediction heads for load, solar, and price forecasting.
- **System-Wide Evaluation**: Provides a consolidated view of performance across all forecasted dimensions.

## Project Structure
- `main.py`: Orchestrates the joint training and evaluation pipeline.
- `data/data_loader.py`: Synchronizes and scales multiple energy data streams.
- `models/joint_model.py`: Multi-stream architecture with shared encoding.
- `evaluation/joint_evaluator.py`: Domain-specific and aggregate metric calculation.

## Usage
Run the joint forecasting pipeline:
```bash
python main.py --hidden_size 128 --epochs 10
```

## Significance
In modern power systems, load, renewable generation, and price are deeply coupled. Jointly forecasting these variables allows for better system-wide optimization, risk management, and market participation strategies.
