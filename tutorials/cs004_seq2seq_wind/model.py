import torch
import torch.nn as nn
from src.models.seq2seq import Encoder, Decoder, Seq2Seq

def build_model(input_size=1, hidden_size=64, num_layers=2, output_size=1, device='cpu'):
    """
    Build the Seq2Seq model with attention for CS004.
    """
    encoder = Encoder(input_size, hidden_size, num_layers)
    decoder = Decoder(output_size, hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model

def train_step(model, src, trg, optimizer, criterion, teacher_forcing_ratio=0.5):
    """
    Perform a single training step.
    """
    model.train()
    optimizer.zero_grad()
    
    trg_len = trg.shape[1]
    outputs = model(src, trg_len, teacher_forcing_ratio, trg)
    
    loss = criterion(outputs, trg)
    loss.backward()
    optimizer.step()
    
    return loss.item()
