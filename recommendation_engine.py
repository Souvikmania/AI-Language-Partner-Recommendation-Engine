import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from preprocess_data import load_and_preprocess_data
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

# Need the model class definition to load the weights
class LanguagePartnerModel(torch.nn.Module):
    def __init__(self):
        super(LanguagePartnerModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def recommend_partners(target_user, all_users_df, model, scaler, top_n=10):
    """
    Predicts compatibility and returns top N recommended partners for a target user.
    """
    # Exclude the target user from potential partners
    potential_partners = all_users_df[all_users_df['user_id'] != target_user['user_id']].copy()
    
    if len(potential_partners) == 0:
        return pd.DataFrame() # No users to compare
        
    # Prepare feature vectors
    # [A_known, A_learning, A_skill, A_streak, B_known, B_learning, B_skill, B_streak]
    
    A_features = {
        'A_known': target_user['known_language_id'],
        'A_learning': target_user['learning_language_id'],
        'A_skill': target_user['skill_level_id'],
        'A_streak': target_user['learning_streak'] # raw streak
    }
    
    num_partners = len(potential_partners)
    
    X_pred = pd.DataFrame(index=range(num_partners))
    X_pred['A_known'] = A_features['A_known']
    X_pred['A_learning'] = A_features['A_learning']
    X_pred['A_skill'] = A_features['A_skill']
    X_pred['A_streak'] = A_features['A_streak']
    
    X_pred['B_known'] = potential_partners['known_language_id'].values
    X_pred['B_learning'] = potential_partners['learning_language_id'].values
    X_pred['B_skill'] = potential_partners['skill_level_id'].values
    X_pred['B_streak'] = potential_partners['learning_streak'].values
    
    streaks = X_pred[['A_streak', 'B_streak']]
    X_pred[['A_streak', 'B_streak']] = scaler.transform(streaks)
    
    X_array = X_pred.values.astype('float32')
    
    # Convert to tensor and predict
    X_tensor = torch.FloatTensor(X_array)
    
    # Ensure model is in eval mode
    model.eval()
    with torch.no_grad():
        probabilities = model(X_tensor).numpy().flatten()
    
    # Add predictions to the potential partners dataset
    potential_partners['compatibility_score'] = probabilities
    
    # Sort and return Top N
    top_matches = potential_partners.sort_values(by='compatibility_score', ascending=False).head(top_n)
    
    return top_matches

if __name__ == "__main__":
    print("Loading files and simulating inference scenario...")
    try:
        # Load synthetic users
        df_users = pd.read_csv('users.csv')
        
        # Load preconfigured Scaler
        _, _, _, _, scaler = load_and_preprocess_data('synthetic_pairs.csv')
        
        # Load Trained Model
        print("Loading Model...")
        model = LanguagePartnerModel()
        model.load_state_dict(torch.load('language_partner_model.pt', weights_only=True))
        
        # Pick a target user
        target_user = df_users.iloc[0].to_dict()
        print("\nTarget User Configuration:")
        print(f"User ID: {target_user['user_id']}")
        print(f"Known Lang ID: {target_user['known_language_id']}, Learning Lang ID: {target_user['learning_language_id']}")
        print(f"Skill: {target_user['skill_level_id']}, Streak: {target_user['learning_streak']}")
        
        print("\nGenerating Top 10 Recommendations...")
        recommendations = recommend_partners(target_user, df_users, model, scaler, top_n=10)
        
        # Print output formatting
        res = recommendations[['user_id', 'known_language_id', 'learning_language_id', 'skill_level_id', 'compatibility_score']]
        print("\n", res.to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"Dependency files missing. Run generator & training first. Error: {e}")
