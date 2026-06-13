import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd
from data_loader import load_data
from feature_engineering import calculate_features, generate_soft_labels, is_complementary_match, is_peer_match
from model import CompatibilityModel
import sys

# Configure UTF-8 encoding for stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr:
            sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Set reproducibility seeds
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_dataset_with_soft_labels(df_users, num_samples=20000):
    """
    Generates training samples with SOFT LABELS for better ranking.
    
    Soft labels allow the model to learn nuanced scoring:
    - Complementary match: 1.0 base
    - Peer match: 0.85 base
    - Adjustments based on streak and skill similarity
    
    Features expanded from 8 to 12:
    Original 8: [A_known, A_learning, A_skill, A_streak, B_known, B_learning, B_skill, B_streak]
    New 4: [streak_diff, skill_diff, is_complementary, is_peer]
    """
    print(f"Preparing Enhanced Dataset from {len(df_users)} users...")
    X, y = [], []
    users = df_users.to_dict('records')
    
    # 1. Natural Random Pairs (~60%)
    print("  Generating natural random pairs...")
    for _ in range(int(num_samples * 0.6)):
        a, b = random.sample(users, 2)
        features = calculate_features(a, b)
        label = generate_soft_labels(a, b)
        X.append(features)
        y.append(label)
        
    # 2. Injected Compatible Pairs (~40% to balance)
    print("  Injecting compatible pairs for ranking learning...")
    for _ in range(int(num_samples * 0.4)):
        a = random.choice(users)
        
        # Try to find a real compatible match
        match_type = random.choice(['comp', 'peer'])
        if match_type == 'comp':
            # Complementary: B_known = A_learning, B_learning = A_known
            b_pool = [u for u in users 
                     if u['known_language'] == a['learning_language'] 
                     and u['learning_language'] == a['known_language'] 
                     and u['user_id'] != a['user_id']]
        else:
            # Peer: B_known = A_known, B_learning = A_learning
            b_pool = [u for u in users 
                     if u['known_language'] == a['known_language'] 
                     and u['learning_language'] == a['learning_language'] 
                     and u['user_id'] != a['user_id']]
            
        if b_pool:
            b = random.choice(b_pool)
        else:
            # Create synthetic match if not found
            b = {
                'user_id': -1,
                'known_language': a['learning_language'] if match_type == 'comp' else a['known_language'],
                'learning_language': a['known_language'] if match_type == 'comp' else a['learning_language'],
                'skill_level': random.choice([1, 2, 3]),
                'streak': random.randint(1, 100)
            }

        features = calculate_features(a, b)
        # Use soft label instead of fixed 1.0
        label = generate_soft_labels(a, b)
        X.append(features)
        y.append(label)
        
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32').reshape(-1, 1)
    
    print(f"  ✓ Dataset has {len(X)} samples with {X.shape[1]} features")
    print(f"  ✓ Label statistics:")
    print(f"      - Mean score: {np.mean(y):.4f}")
    print(f"      - Max score: {np.max(y):.4f}")
    print(f"      - Min score: {np.min(y):.4f}")
    
    return X, y

def run_training(epochs=30, batch_size=32):
    set_seeds()
    print("\n" + "="*70)
    print("TRAINING ENHANCED AI RANKING MODEL WITH SOFT LABELS")
    print("="*70)
    
    df = load_data()
    print(f"Loaded {len(df)} users.")
    
    X, y = prepare_dataset_with_soft_labels(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # DataLoader setup
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model with 12 input features
    model = CompatibilityModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()  # MSE for regression-like soft label prediction
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nModel Architecture (12-feature enhanced):")
    print(model)
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)
    
    # Training loop
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch+1) % max(1, epochs//5) == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.6f}")
            
    print("-" * 70)
    print("Training completed. Evaluating on test set...\n")
    
    # Evaluation (treating as regression)
    model.eval()
    with torch.no_grad():
        y_pred_soft = model(torch.tensor(X_test)).numpy()
        y_pred_binary = (y_pred_soft > 0.5).astype(int)
        
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred_soft)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred_soft))
    
    # Classification metrics (thresholded at 0.5)
    acc = accuracy_score(y_test > 0.5, y_pred_binary)
    prec = precision_score(y_test > 0.5, y_pred_binary, zero_division=0)
    rec = recall_score(y_test > 0.5, y_pred_binary, zero_division=0)
    
    print(f"================ REGRESSION METRICS (Soft Labels) ===============")
    print(f"Mean Squared Error:   {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error:  {mae:.6f}")
    print()
    print(f"================ CLASSIFICATION METRICS (Thresholded) ============")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("================================================================\n")
    
    # Show score distribution
    print(f"================ PREDICTED SCORE DISTRIBUTION ==================")
    print(f"Mean predicted score: {np.mean(y_pred_soft):.4f}")
    print(f"Score range: [{np.min(y_pred_soft):.4f}, {np.max(y_pred_soft):.4f}]")
    print(f"Standard deviation: {np.std(y_pred_soft):.4f}")
    print("================================================================\n")
    
    # Save model
    torch.save(model.state_dict(), 'compatibility_model.pt')
    print("✓ Model weights saved to compatibility_model.pt")
    
    return model

if __name__ == "__main__":
    run_training()
