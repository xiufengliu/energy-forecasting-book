import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('manuscript/code/src'))

def test_models():
    # Test LSTM
    from models.rnn import LSTMModel
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    x = torch.randn(32, 24, 1)
    y = model(x)
    assert y.shape == (32, 1)
    print("LSTM check passed.")

    # Test Transformer
    from models.transformer import TransformerModel
    model = TransformerModel(input_size=1, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=64, output_size=1)
    # Transformer expects (seq_len, batch_size, input_size)
    x = torch.randn(24, 32, 1)
    y = model(x)
    assert y.shape == (32, 1)
    print("Transformer check passed.")

    # Test TCN
    from models.cnn import TCNModel
    model = TCNModel(input_size=1, num_channels=[16, 32, 64], kernel_size=3)
    x = torch.randn(32, 24, 1)
    y = model(x)
    assert y.shape == (32, 1)
    print("TCN check passed.")

    # Test Probabilistic
    from models.probabilistic import ProbabilisticLSTM, QuantileLoss
    model = ProbabilisticLSTM(input_size=1, hidden_size=64, num_layers=2, num_quantiles=3)
    x = torch.randn(32, 24, 1)
    y = model(x)
    assert y.shape == (32, 3)
    print("Probabilistic model check passed.")
    
    criterion = QuantileLoss([0.1, 0.5, 0.9])
    target = torch.randn(32, 1)
    loss = criterion(y, target)
    assert loss.item() > 0
    print("Quantile Loss check passed.")

if __name__ == "__main__":
    try:
        test_models()
        print("\nAll core models verified successfully.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
