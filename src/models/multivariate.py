import torch
import torch.nn as nn

class ChannelIndependentWrapper(nn.Module):
    """
    Wrapper for channel-independent multivariate forecasting.
    Treats each variable as a separate time series.
    """
    def __init__(self, base_model, num_channels):
        super(ChannelIndependentWrapper, self).__init__()
        self.num_channels = num_channels
        self.models = nn.ModuleList([base_model for _ in range(num_channels)])

    def forward(self, x):
        # x: (batch, seq_len, num_channels)
        outputs = []
        for i in range(self.num_channels):
            # Take one channel, add feature dimension
            channel_input = x[:, :, i].unsqueeze(-1)
            outputs.append(self.models[i](channel_input))
        
        return torch.stack(outputs, dim=-1) # (batch, output_dim, num_channels)

class ChannelDependentWrapper(nn.Module):
    """
    Wrapper for channel-dependent multivariate forecasting.
    Expects the base model to handle multiple input channels.
    """
    def __init__(self, base_model, input_size, output_size):
        super(ChannelDependentWrapper, self).__init__()
        self.model = base_model
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out = self.model(x)
        return out

class MultivariateFeatureExtractor(nn.Module):
    """
    Extracts features from multivariate time series.
    """
    def __init__(self, input_size, hidden_size):
        super(MultivariateFeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2) # (batch, input_size, seq_len)
        features = self.conv(x)
        features = self.pool(features).squeeze(-1)
        return features
