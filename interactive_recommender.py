import pandas as pd
import numpy as np
import torch
import sys
import os
from recommendation_engine import LanguagePartnerModel, recommend_partners
from preprocess_data import load_and_preprocess_data

# Configure UTF-8 encoding for stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr:
            sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Language Dictionary encoding
LANGUAGES = {
    'English': 1, 'Spanish': 2, 'French': 3, 'German': 4, 'Hindi': 5,
    'Japanese': 6, 'Korean': 7, 'Chinese': 8, 'Italian': 9, 'Arabic': 10
}
ID_TO_LANG = {v: k for k, v in LANGUAGES.items()}

# Skill Level encoding
SKILL_LEVELS = {
    'Beginner': 1, 'Intermediate': 2, 'Advanced': 3
}
ID_TO_SKILL = {v: k for k, v in SKILL_LEVELS.items()}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60 + "\n")

def get_input(prompt, options=None, is_int=False):
    while True:
        val = input(prompt).strip()
        if not val:
            continue
        if options:
            if val.lower() in [o.lower() for o in options]:
                # Find the original case option
                for o in options:
                    if o.lower() == val.lower():
                        return o
            print(f"Invalid option. Please choose from: {', '.join(options)}")
        elif is_int:
            try:
                return int(val)
            except ValueError:
                print("Please enter a valid number.")
        else:
            return val

def main():
    # Load files
    try:
        df_users = pd.read_csv('users.csv')
        _, _, _, _, scaler = load_and_preprocess_data('synthetic_pairs.csv')
        
        model = LanguagePartnerModel()
        model.load_state_dict(torch.load('language_partner_model.pt', weights_only=True))
        model.eval()
    except Exception as e:
        print(f"Error loading dependencies: {e}. Please run generate_dataset.py and train_model.py first.")
        return

    while True:
        clear_screen()
        print_header("LANGUAGE PARTNER AI - INTERACTIVE MODE")
        print("1. Find matches for an EXISTING User (by ID)")
        print("2. Find matches for a CUSTOM Profile (Enter your own)")
        print("3. Exit")
        
        choice = get_input("\nSelect an option (1-3): ", ["1", "2", "3"])
        
        if choice == "3":
            print("\nGoodbye!")
            break
            
        target_user = None
        
        if choice == "1":
            user_id = get_input("Enter User ID (1-1000): ", is_int=True)
            user_data = df_users[df_users['user_id'] == user_id]
            if user_data.empty:
                print("User ID not found.")
                input("\nPress Enter to continue...")
                continue
            target_user = user_data.iloc[0].to_dict()
        else:
            print("\n--- Enter Your Profile Details ---")
            known = get_input(f"Your native language ({', '.join(LANGUAGES.keys())}): ", list(LANGUAGES.keys()))
            learning = get_input(f"Language you want to learn: ", [l for l in LANGUAGES.keys() if l != known])
            skill = get_input(f"Your current skill level ({', '.join(SKILL_LEVELS.keys())}): ", list(SKILL_LEVELS.keys()))
            streak = get_input("Your current learning streak (days): ", is_int=True)
            
            target_user = {
                'user_id': 9999, # Dummy ID
                'known_language_id': LANGUAGES[known],
                'learning_language_id': LANGUAGES[learning],
                'skill_level_id': SKILL_LEVELS[skill],
                'learning_streak': streak
            }

        clear_screen()
        print_header("SEARCHING FOR THE PERFECT PARTNER...")
        print(f"Profile: Native {ID_TO_LANG[target_user['known_language_id']]}, Learning {ID_TO_LANG[target_user['learning_language_id']]}")
        print(f"Skill: {ID_TO_SKILL[target_user['skill_level_id']]}, Streak: {target_user['learning_streak']} days")
        
        recommendations = recommend_partners(target_user, df_users, model, scaler, top_n=10)
        
        print("\n" + "-"*85)
        print(f"{'Rank':<5} | {'Partner ID':<10} | {'Speaks':<12} | {'Learning':<12} | {'Skill':<12} | {'Compatibility'}")
        print("-"*85)
        
        for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
            lang_name = ID_TO_LANG[int(row['known_language_id'])]
            learning_name = ID_TO_LANG[int(row['learning_language_id'])]
            skill_name = ID_TO_SKILL[int(row['skill_level_id'])]
            score = row['compatibility_score']
            
            print(f"{i:<5} | {int(row['user_id']):<10} | {lang_name:<12} | {learning_name:<12} | {skill_name:<12} | {score:,.4f}")
            
        print("-"*85)
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
