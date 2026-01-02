from src.models.cnn import CNNModel # Assuming TCN/CNN is here
from src.models.transformer import TransformerModel

def get_model(model_type, input_size, hidden_size, num_layers):
    """
    Instantiate model for renewable forecasting.
    """
    if model_type == "tcn":
        # Simplified TCN wrapper
        return CNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    elif model_type == "transformer":
        return TransformerModel(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=num_layers,
            dim_feedforward=128,
            output_size=1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
