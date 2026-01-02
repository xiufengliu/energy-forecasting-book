from src.models.faformer import FAformerModel
from src.models.transformer import TransformerModel

def get_model(model_type, input_size, hidden_size, num_layers):
    """
    Instantiate model for price forecasting.
    """
    if model_type == "faformer":
        return FAformerModel(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=num_layers,
            dim_feedforward=128,
            output_size=1,
            top_k=10
        )
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
