import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x):
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        self.output_size = output_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
        if input_size != output_size:
            self.res_fc = nn.Linear(input_size, output_size)
        else:
            self.res_fc = nn.Identity()

    def forward(self, x, context=None):
        out = self.fc1(x)
        if context is not None and self.context_size:
            out = out + self.context_fc(context)
        out = F.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        
        res = self.res_fc(x)
        return self.layer_norm(out + res)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_vars, hidden_size, dropout=0.1, context_size=None):
        super(VariableSelectionNetwork, self).__init__()
        self.num_vars = num_vars
        self.grns = nn.ModuleList([GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout) for _ in range(num_vars)])
        self.selector_grn = GatedResidualNetwork(num_vars * input_size, hidden_size, num_vars, dropout, context_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        # x: (batch, num_vars, input_size)
        var_outputs = []
        for i in range(self.num_vars):
            var_outputs.append(self.grns[i](x[:, i, :]))
        var_outputs = torch.stack(var_outputs, dim=1) # (batch, num_vars, hidden_size)
        
        flattened_x = x.view(x.size(0), -1)
        weights = self.selector_grn(flattened_x, context)
        weights = self.softmax(weights).unsqueeze(2) # (batch, num_vars, 1)
        
        selected_output = torch.sum(weights * var_outputs, dim=1)
        return selected_output, weights

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        
        self.q_fc = nn.Linear(d_model, d_model)
        self.k_fc = nn.Linear(d_model, d_model)
        self.v_fc = nn.Linear(d_model, d_model)
        self.output_fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        
        # Linear projection
        q = self.q_fc(q).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_fc(k).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_fc(v).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.output_fc(context), attn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super(TemporalFusionTransformer, self).__init__()
        # Simplified configuration for demonstration
        self.d_model = config['d_model']
        self.num_vars = config['num_vars']
        
        self.vsn = VariableSelectionNetwork(config['input_size'], self.num_vars, self.d_model)
        self.lstm = nn.LSTM(self.d_model, self.d_model, batch_first=True)
        self.attn = InterpretableMultiHeadAttention(config['n_head'], self.d_model)
        self.fc_out = nn.Linear(self.d_model, config['output_size'])

    def forward(self, x):
        # x: (batch, seq_len, num_vars, input_size)
        batch_size, seq_len, num_vars, input_size = x.size()
        
        # Apply VSN across time steps
        vsn_outputs = []
        for t in range(seq_len):
            out, _ = self.vsn(x[:, t, :, :])
            vsn_outputs.append(out)
        vsn_outputs = torch.stack(vsn_outputs, dim=1) # (batch, seq_len, d_model)
        
        # Temporal processing with LSTM
        lstm_out, _ = self.lstm(vsn_outputs)
        
        # Self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        
        # Final prediction
        output = self.fc_out(attn_out[:, -1, :])
        return output
