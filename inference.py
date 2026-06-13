import torch
import pandas as pd
from model import CompatibilityModel
from feature_engineering import calculate_features
from data_loader import load_data
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

def run_ml_inference():
    print("\n" + "="*115)
    print(f"{'AI LANGUAGE PARTNER RECOMMENDATION SYSTEM (v2.0)':^115}")
    print(f"{'MODEL MATCHING CRITERIA: COMPLEMENTARY EXCHANGE & PEER LEARNING':^115}")
    print("="*115 + "\n")
    
    print("Match Logic Explained:")
    print("1. COMPLEMENTARY EXCHANGE: Match where both users want to learn each other's native language.")
    print("2. PEER LEARNING: Match where both users share the same native and target languages.")
    print("-" * 115 + "\n")
    
    # User Inputs (ONLY Your Profile)
    print("--- Tell the AI About Your Languages ---")
    my_native = input("What language do you speak natively? ").strip()
    my_target = input("What language are you trying to learn? ").strip()
    
    # Create target user object
    target_user = {
        'user_id': -1,
        'known_language': my_native,
        'learning_language': my_target,
        'skill_level': 1, # Not used for matching
        'streak': 1       # Not used for matching
    }

    # Load search database & ML model
    df_users = load_data()
    others = df_users.to_dict('records')
    model = CompatibilityModel()
    
    # Check if model exists
    try:
        model.load_state_dict(torch.load('compatibility_model.pt', weights_only=True))
    except FileNotFoundError:
        print("Model weights not found. Running training first...")
        import subprocess
        subprocess.run(["python", "train.py"])
        model.load_state_dict(torch.load('compatibility_model.pt', weights_only=True))
    model.eval()

    print(f"\nAI is scanning the 10,000-user database to find the best matches for a {my_native.title()} speaker learning {my_target.title()}...")

    # Batch compute ML features for EVERY user in the database
    features_list = []
    for user_j in others:
        features = calculate_features(target_user, user_j)
        features_list.append(features)

    # ML Model Suggestion (Prediction)
    X_tensor = torch.tensor(features_list, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_tensor).numpy().flatten()

    df_results = df_users.copy()
    df_results['compatibility_score'] = scores
    
    # Match Type Labeling
    def get_match_label(user_i, user_j):
        # Native Exchange (Complementary)
        if user_i['learning_language'].upper() == user_j['known_language'].upper() and \
           user_j['learning_language'].upper() == user_i['known_language'].upper():
            return "COMPLEMENTARY (Exchange)"
        
        # Peer Learning
        if user_i['known_language'].upper() == user_j['known_language'].upper() and \
           user_i['learning_language'].upper() == user_j['learning_language'].upper():
            return "PEER LEARNING (Goal Match)"
            
        return "GENERAL MATCH"
        
    df_results['match_type'] = df_results.apply(
        lambda row: get_match_label(target_user, row.to_dict()), axis=1
    )

    # Filter out anything with very low score (below 0.1)
    # Then sort by compatibility score
    top_suggestions = df_results[df_results['compatibility_score'] > 0.1].sort_values(by='compatibility_score', ascending=False).head(10)

    if top_suggestions.empty:
        print(f"\nNo exact matches found in the database for a {my_native.title()} speaker learning {my_target.title()}.")
        print("Model suggests expanding search criteria or checking for typos.")
    else:
        print(f"\nML Model found {len(top_suggestions)} ideal language matches: \n")
        print(f"{'user_id':<8} | {'score':<6} | {'match_description':<35} | {'known_lang':<15} | {'learn_lang':<15}")
        print("-" * 105)
        
        for _, row in top_suggestions.iterrows():
            print(f"{int(row['user_id']):<8} | {row['compatibility_score']:.2f} | {row['match_type']:<35} | {row['known_language']:<15} | {row['learning_language']:<15}")
        print("-" * 105 + "\n")

if __name__ == "__main__":
    run_ml_inference()
