import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath='synthetic_pairs.csv'):
    """
    Loads dataset, scales continuous features (streaks), and splits into train/val.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found. Run generate_dataset.py first.")
        
    print("Preprocessing data...")
    # Features (X) and Labels (y)
    feature_cols = [
        'A_known', 'A_learning', 'A_skill', 'A_streak',
        'B_known', 'B_learning', 'B_skill', 'B_streak'
    ]
    
    X = df[feature_cols].copy()
    y = df['label'].values
    
    # Scale continuous features: streaks
    # The streaks are between 1 and 100, we want them scaled 0 to 1
    # Note: Language and skill values are considered ordinal/categorical
    # For a simple dense network they can be fed as-is if properly bounded,
    # or one-hot encoded but the instruction requests single numerical values.
    
    # Let's scale streaks explicitly
    scaler = MinMaxScaler()
    X[['A_streak', 'B_streak']] = scaler.fit_transform(X[['A_streak', 'B_streak']])
    
    # Additional: optionally scale categorical variables to 0-1 for better NN convergence, 
    # but since they're heavily mapped 1-10 and 1-3, standardizing/normalizing might help.
    # Instruction specifies to "Scale streak values between 0 and 1", and "Ensure categorical features are encoded numerically." 
    # (they already are 1-10, 1-3). We will convert the DataFrame to a NumPy array.
    
    X_array = X.values.astype('float32')
    y_array = y.astype('float32')
    
    # Train/Validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_array, y_array, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_array  # Maintain the same balance of 0s and 1s
    )
    
    print(f"Preprocessing complete. Train size: {len(X_train)}, Val size: {len(X_val)}")
    return X_train, X_val, y_train, y_val, scaler

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess_data()
    print("Sample preprocessed input (X_train[0]):", X_train[0])
    print("Sample label (y_train[0]):", y_train[0])
