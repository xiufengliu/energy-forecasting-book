# Deep Learning for Energy Forecasting: Code & Tutorials

This repository contains the complete collection of code, tutorials, and working examples for the book **"Deep Learning for Energy Forecasting: A Practical Guide"**. 

## Repository Structure

The code is organized to mirror the book's structure, focusing on foundation models, advanced architectures, and real-world application pipelines.

- `src/`: Core model architectures and utility functions.
  - `models/`: RNN, CNN, Transformer, Seq2Seq, TFT, FAformer, etc.
  - `utils/`: Metrics, plotting, and training helpers.
- `tutorials/`: Step-by-step Jupyter notebooks for each chapter.
  - `cs001-cs006`: Core modeling techniques (RNN, CNN, Seq2Seq, Probabilistic).
  - `cs009-013`: Advanced optimization and Transformer applications.
- `case_studies/`: End-to-end industry application pipelines.
  - `cs007-008`: Load and Renewable forecasting pipelines.
  - `cs014-016`: Joint forecasting and value-oriented implementations.
- `data/`: Sample data and data downloading scripts.
- `tests/`: Automated test suites for model verification.

## Key Case Studies

| ID | Title | Concept |
|----|-------|---------|
| CS004 | Seq2Seq Wind | Encoder-Decoder + Attention |
| CS007 | E2E Load Pipeline | Production MLOps Workflow |
| CS008 | E2E Renewable | NWP Integration & WAPE |
| CS012 | TFT Price | Temporal Fusion Transformers |
| CS013 | FAformer | Frequency-Aware Attention |
| CS016 | Value-Oriented | Profit-Aware Optimization |

## Setup and Installation

All examples are implemented in Python using PyTorch.

```bash
# Clone the repository
git clone https://github.com/xiufengliu/energy-forecasting-book.git
cd energy-forecasting-book/manuscript/code

# Install dependencies
pip install -r requirements.txt
```

## How to Use This Repository
For each chapter in the book, you will find a corresponding tutorial in the `tutorials/` directory. We recommend running the tutorials in order to build your foundation before exploring the end-to-end case studies.

## License
This code is released under the MIT License.
