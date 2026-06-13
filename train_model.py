import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Import the data preprocessing pipeline
from preprocess_data import load_and_preprocess_data

class LanguagePartnerModel(nn.Module):
    def __init__(self):
        super(LanguagePartnerModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(epochs=20, batch_size=32):
    """
    Trains the model and evaluates it using Test/Val sets
    """
    print("Loading datasets...")
    # Load standardized testing sets
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess_data('synthetic_pairs.csv')
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Building model architecture...")
    model = LanguagePartnerModel()
    print(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    print(f"Starting training loop... (Epochs={epochs}, BatchSize={batch_size})")
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy track during training
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')
    
    print("\nTraining completed. Evaluating on Validation Set...")
    # Evaluation Mode
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        y_pred_probs = val_outputs.numpy()
        
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs > 0.5).astype(int)
    y_true = y_val_tensor.numpy()
    
    # Evaluation Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n================ EVALUATION METRICS ================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("====================================================\n")
    
    # Save the model
    model_path = 'language_partner_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to '{model_path}'")

if __name__ == "__main__":
    train_and_evaluate()
