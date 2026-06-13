import pandas as pd
import random
import os

LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Hindi', 'Japanese', 'Chinese', 'Italian', 'Arabic', 'Korean', 'Bengali']
SKILL_LEVELS = [1, 2, 3]  # Numeric IDs for skills

def generate_synthetic_csv(filename='users.csv', num_users=1000):
    """Generates a synthetic users.csv with consistent column names."""
    print(f"Generating synthetic dataset: {num_users} users...")
    users = []
    for user_id in range(1, num_users + 1):
        known = random.choice(LANGUAGES)
        learning = random.choice([l for l in LANGUAGES if l != known])
        skill = random.choice(SKILL_LEVELS)
        streak = random.randint(1, 100)
        
        users.append([user_id, known, learning, skill, streak])
    
    df = pd.DataFrame(users, columns=['user_id', 'known_language', 'learning_language', 'skill_level', 'streak'])
    df.to_csv(filename, index=False)
    print(f"Successfully saved to {filename}")
    return df

def load_data(filename='users.csv'):
    """Loads user data from CSV."""
    if not os.path.exists(filename):
        return generate_synthetic_csv(filename)
    return pd.read_csv(filename)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
