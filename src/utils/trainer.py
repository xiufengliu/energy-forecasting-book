import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class Trainer:
    """Standard trainer for forecasting models."""
    def __init__(self, model, criterion, learning_rate=1e-3, device='cpu', patience=10):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).float()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, save_path='best_model.pth'):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))
        return self.model

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions)
