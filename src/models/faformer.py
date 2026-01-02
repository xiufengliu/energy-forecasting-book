import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyAwareAttention(nn.Module):
    def __init__(self, d_model, top_k=10):
        super(FrequencyAwareAttention, self).__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        
        # Fast Fourier Transform
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Get amplitudes
        amplitudes = torch.abs(x_fft)
        
        # Find top-k frequencies
        # Sum amplitudes across d_model dimension to find dominant frequencies for the whole signal
        mean_amplitudes = torch.mean(amplitudes, dim=2)
        _, top_indices = torch.topk(mean_amplitudes, self.top_k, dim=1)
        
        # Filter FFT results to keep only top-k frequencies
        # Create a mask for top-k frequencies
        mask = torch.zeros_like(x_fft)
        for i in range(batch_size):
            mask[i, top_indices[i], :] = 1
            
        x_fft_filtered = x_fft * mask
        
        # Inverse Fast Fourier Transform
        x_ifft = torch.fft.irfft(x_fft_filtered, n=seq_len, dim=1)
        
        return self.fc(x_ifft)

class FAformerModel(nn.Module):
    """Frequency-Aware Transformer for energy forecasting."""
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, top_k=10, dropout=0.1):
        super(FAformerModel, self).__init__()
        self.encoder_input = nn.Linear(input_size, d_model)
        self.fa_attn = FrequencyAwareAttention(d_model, top_k)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.encoder_input(x)
        
        # Frequency-aware attention branch
        fa_out = self.fa_attn(x)
        
        # Standard transformer branch
        trans_out = self.transformer_encoder(x)
        
        # Combine branches (simple addition)
        combined = fa_out + trans_out
        
        # Prediction
        output = self.decoder(combined[:, -1, :])
        return output
