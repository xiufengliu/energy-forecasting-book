import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, (hidden, cell) = self.rnn(x)
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state seq_len times
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Energy calculation
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = Attention(hidden_size)
        self.rnn = nn.LSTM(hidden_size + 1, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2 + 1, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: (batch, 1) - last prediction or actual
        # hidden: (num_layers, batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        
        # Get attention weights
        # Use only the top layer hidden state for attention
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1) # (batch, 1, seq_len)
        
        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs) # (batch, 1, hidden_size)
        
        rnn_input = torch.cat((input.unsqueeze(1), weighted), dim=2) # (batch, 1, hidden_size + 1)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Concatenate RNN output, weighted context, and input for final prediction
        prediction = self.fc_out(torch.cat((output, weighted, input.unsqueeze(1)), dim=2))
        
        return prediction.squeeze(1), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg_len, teacher_forcing_ratio=0.5, trg=None):
        # src: (batch, seq_len, input_size)
        batch_size = src.shape[0]
        output_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, output_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Start with the last value of the input sequence
        input = src[:, -1, 0].unsqueeze(1) # Simple assumption: first feature is the target
        
        for t in range(trg_len):
            prediction, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force and trg is not None:
                input = trg[:, t, 0].unsqueeze(1)
            else:
                input = prediction[:, 0].unsqueeze(1) # Feed prediction back
                
        return outputs
