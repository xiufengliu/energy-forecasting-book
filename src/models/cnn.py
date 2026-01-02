import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=self.padding, dilation=dilation))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        # Chausal convolution: remove future context from the output
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return self.relu(out)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, stride, dilation)
        self.dropout2 = nn.Dropout(dropout)
        
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        return self.relu(out + residual)

class TCNModel(nn.Module):
    """Temporal Convolutional Network (TCN) for energy forecasting."""
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, 
                               dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)
        # x shape: (batch_size, input_size, seq_len)
        y1 = self.network(x)
        # y1 shape: (batch_size, out_channels, seq_len)
        return self.fc(y1[:, :, -1])
