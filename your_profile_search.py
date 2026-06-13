import torch
import numpy as np
import pandas as pd
import sys
from model import CompatibilityModel
from feature_engineering import calculate_features
from data_loader import load_data

# Configure UTF-8 encoding for stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr:
            sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def interactive_search():
    print("\n" + "="*80)
    print(f"{'CUSTOM PROFILE SEARCH':^80}")
    print("="*80 + "\n")
    
    print("The AI is waiting for your profile to provide personalized recommendations.\n")
    
    # User Inputs
    print("Enter your details below:")
    known_lang = input("What language do you speak natively? ").strip()
    learn_lang = input("What language are you trying to learn? ").strip()
    
    try:
        skill = int(input("What is your current skill level (1=Beg, 2=Int, 3=Adv)? "))
        streak = int(input("How many days is your current streak? "))
    except ValueError:
        print("Invalid input for skill or streak. Please enter numbers.")
        return

    # Create target user object
    target_user = {
        'user_id': -1, # Temporary ID
        'known_language': known_lang,
        'learning_language': learn_lang,
        'skill_level': skill,
        'streak': streak
    }

    # Load search database
    df_users = load_data()
    others = df_users.to_dict('records')

    # Load model
    model = CompatibilityModel()
    try:
        model.load_state_dict(torch.load('compatibility_model.pt', weights_only=True))
    except FileNotFoundError:
        print("Model file not found. Please run main.py first to train.")
        return
    model.eval()

    print(f"\nSearching database for the best partners for a {known_lang} speaker learning {learn_lang}...")

    # Batch compute features
    features_list = []
    for user_j in others:
        features = calculate_features(target_user, user_j)
        features_list.append(features)

    # Predict
    X_tensor = torch.tensor(features_list, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_tensor).numpy().flatten()

    df_users['predicted_score'] = scores
    
    # Formatting
    def determine_match_type(user_i, user_j):
        if user_i['learning_language'] == user_j['known_language'] and \
           user_j['learning_language'] == user_i['known_language']:
            return "Complementary"
        if user_i['known_language'] == user_j['known_language'] and \
           user_i['learning_language'] == user_j['learning_language']:
            return "Peer"
        return "Broad"
        
    df_users['match_type'] = df_users.apply(lambda row: determine_match_type(target_user, row.to_dict()), axis=1)

    top_n = df_users.sort_values(by='predicted_score', ascending=False).head(10)

    print(f"\nTop Recommendations for YOU (New User):\n")
    print(f"{'user_id':<8} | {'score':<6} | {'match_type':<15} | {'known_lang':<15} | {'learn_lang':<15} | {'streak':<8} | {'skill'}")
    print("-" * 105)
    
    for _, row in top_n.iterrows():
        print(f"{int(row['user_id']):<8} | {row['predicted_score']:.2f} | {row['match_type']:<15} | {row['known_language']:<15} | {row['learning_language']:<15} | {int(row['streak']):<8} | {int(row['skill_level'])}")
    print("-" * 105 + "\n")

if __name__ == "__main__":
    interactive_search()
