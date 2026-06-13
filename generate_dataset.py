import numpy as np
import pandas as pd
import random

# Language Dictionary encoding
LANGUAGES = {
    'English': 1,
    'Spanish': 2,
    'French': 3,
    'German': 4,
    'Hindi': 5,
    'Japanese': 6,
    'Chinese': 7,
    'Italian': 8,
    'Arabic': 9,
    'Korean': 10,
    'Bengali': 11
}

# Skill Level encoding
SKILL_LEVELS = {
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3
}

def generate_users(num_users=1000):
    users = []
    for user_id in range(1, num_users + 1):
        known_lang_name, known_lang_id = random.choice(list(LANGUAGES.items()))
        # Ensure learning language is different from known language
        learning_langs_pool = {k: v for k, v in LANGUAGES.items() if k != known_lang_name}
        learning_lang_name, learning_lang_id = random.choice(list(learning_langs_pool.items()))
        
        skill_name, skill_id = random.choice(list(SKILL_LEVELS.items()))
        streak = random.randint(1, 100)
        
        users.append({
            'user_id': user_id,
            'known_language_id': known_lang_id,
            'learning_language_id': learning_lang_id,
            'skill_level_id': skill_id,
            'learning_streak': streak
        })
    return pd.DataFrame(users)

def is_compatible(user_a, user_b):
    # Case 1: Complementary Language Exchange
    if user_a['learning_language_id'] == user_b['known_language_id'] and \
       user_b['learning_language_id'] == user_a['known_language_id']:
        return 1
    
    # Case 2: Peer Learning Match
    if user_a['known_language_id'] == user_b['known_language_id'] and \
       user_a['learning_language_id'] == user_b['learning_language_id']:
        return 1
        
    return 0

def generate_training_pairs(df_users, num_pairs=20000):
    pairs = []
    users = df_users.to_dict('records')
    
    # Generating purely random pairs.
    # To avoid having too few compatible pairs (since it's a rare event randomly),
    # let's try to oversample compatible pairs intentionally.
    compatible_pairs = []
    incompatible_pairs = []
    
    # 1. Random pairs approach
    for _ in range(num_pairs * 2): # Generate more to filter
        user_a, user_b = random.sample(users, 2)
        
        label = is_compatible(user_a, user_b)
        
        pair = {
            'A_known': user_a['known_language_id'],
            'A_learning': user_a['learning_language_id'],
            'A_skill': user_a['skill_level_id'],
            'A_streak': user_a['learning_streak'],
            'B_known': user_b['known_language_id'],
            'B_learning': user_b['learning_language_id'],
            'B_skill': user_b['skill_level_id'],
            'B_streak': user_b['learning_streak'],
            'label': label
        }
        
        if label == 1:
            compatible_pairs.append(pair)
        else:
            incompatible_pairs.append(pair)
            
        if len(compatible_pairs) + len(incompatible_pairs) >= num_pairs and len(compatible_pairs) > (num_pairs * 0.1): 
            # early exit if we have enough and at least 10% compatible
            break

    # If we didn't get enough compatible pairs naturally (very likely with 10 classes), 
    # we need to artificially inject some purely matching pairs to balance dataset.
    num_to_inject = max(0, int(num_pairs * 0.4) - len(compatible_pairs))
    for _ in range(num_to_inject):
        user_a = random.choice(users)
        
        # Determine match type randomly
        match_type = random.choice([1, 2])
        if match_type == 1:
            # Complementary Exchange match
            target_known = user_a['learning_language_id']
            target_learning = user_a['known_language_id']
        else:
            # Peer Learning match
            target_known = user_a['known_language_id']
            target_learning = user_a['learning_language_id']
            
        # Try to find a real user that matches, or just create a synthetic B user for balancing
        potential_matches = [u for u in users if u['known_language_id'] == target_known and u['learning_language_id'] == target_learning and u['user_id'] != user_a['user_id']]
        
        if potential_matches:
            user_b = random.choice(potential_matches)
        else:
            # Generate dummy user B to satisfy the condition constraint heavily
            user_b = {
                'user_id': -1, # sentinel value
                'known_language_id': target_known,
                'learning_language_id': target_learning,
                'skill_level_id': random.randint(1, 3),
                'learning_streak': random.randint(1, 100)
            }
            
        pair = {
            'A_known': user_a['known_language_id'],
            'A_learning': user_a['learning_language_id'],
            'A_skill': user_a['skill_level_id'],
            'A_streak': user_a['learning_streak'],
            'B_known': user_b['known_language_id'],
            'B_learning': user_b['learning_language_id'],
            'B_skill': user_b['skill_level_id'],
            'B_streak': user_b['learning_streak'],
            'label': 1
        }
        compatible_pairs.append(pair)
        
    # Combine and shuffle
    target_incompatible = num_pairs - len(compatible_pairs)
    final_pairs = compatible_pairs + incompatible_pairs[:target_incompatible]
    random.shuffle(final_pairs)
    
    return pd.DataFrame(final_pairs[:num_pairs])

if __name__ == "__main__":
    print("Generating synthetic user dataset...")
    df_users = generate_users(1000)
    df_users.to_csv('users.csv', index=False)
    print(f"Generated {len(df_users)} users and saved to users.csv")
    
    print("Generating training pairs dataset...")
    df_pairs = generate_training_pairs(df_users, 20000)
    df_pairs.to_csv('synthetic_pairs.csv', index=False)
    
    label_counts = df_pairs['label'].value_counts()
    print(f"Generated {len(df_pairs)} training pairs and saved to synthetic_pairs.csv")
    print(f"Label distribution:\n{label_counts}")
