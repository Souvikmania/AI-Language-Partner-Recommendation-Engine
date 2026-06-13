import torch
import numpy as np
import pandas as pd
from data_loader import load_data
from feature_engineering import calculate_features
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

def recommend_partners(target_user_id, top_n=10):
    df_users = load_data()
    target_user_all = df_users[df_users['user_id'] == target_user_id]
    
    if target_user_all.empty:
        print(f"User ID {target_user_id} not found.")
        return None
    
    target_user = target_user_all.iloc[0].to_dict()
    other_users = df_users[df_users['user_id'] != target_user_id].copy()
    
    # Load model
    model = CompatibilityModel()
    try:
        model.load_state_dict(torch.load('compatibility_model.pt', weights_only=True))
    except FileNotFoundError:
        print("Error: Trained model weights not found. Run train.py first.")
        return None
        
    model.eval()
    
    print(f"\nSearching for the Perfect Partners for User {target_user_id}...")
    print(f"Native: {target_user['known_language']}, Learning: {target_user['learning_language']}")
    
    features_list = []
    ids_list = []
    
    for _, user_j in other_users.iterrows():
        features = calculate_features(target_user, user_j.to_dict())
        features_list.append(features)
        ids_list.append(user_j['user_id'])
        
    # Batch predict
    X_tensor = torch.tensor(features_list, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_tensor).numpy().flatten()
        
    other_users['predicted_score'] = scores
    
    # Sort and rank
    recommendations = other_users.sort_values(by='predicted_score', ascending=False).head(top_n)
    
    # Format Match Type (Optional logic for display)
    def determine_match_type(user_i, user_j):
        if user_i['learning_language'] == user_j['known_language'] and \
           user_j['learning_language'] == user_i['known_language']:
            return "Complementary"
        if user_i['known_language'] == user_j['known_language'] and \
           user_i['learning_language'] == user_j['learning_language']:
            return "Peer"
        return "Broad"
        
    recommendations['match_type'] = recommendations.apply(lambda row: determine_match_type(target_user, row.to_dict()), axis=1)
    
    return recommendations[['user_id', 'predicted_score', 'match_type', 'known_language', 'learning_language', 'streak', 'skill_level']]

def display_recommendations(target_user_id, recommendations):
    if recommendations is None: return
    
    print(f"\nTop Recommendations for User {target_user_id}:\n")
    print(f"{'user_id':<8} | {'score':<6} | {'match_type':<15} | {'known_lang':<15} | {'learn_lang':<15} | {'streak':<8} | {'skill'}")
    print("-" * 105)
    
    for _, row in recommendations.iterrows():
        print(f"{int(row['user_id']):<8} | {row['predicted_score']:.2f} | {row['match_type']:<15} | {row['known_language']:<15} | {row['learning_language']:<15} | {int(row['streak']):<8} | {int(row['skill_level'])}")
    print("-" * 105 + "\n")

if __name__ == "__main__":
    recs = recommend_partners(1)
    display_recommendations(1, recs)
