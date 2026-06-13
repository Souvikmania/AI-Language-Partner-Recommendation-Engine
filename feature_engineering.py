import pandas as pd
import numpy as np

# Language mapping (string name to numeric ID)
LANGUAGE_MAP = {
    'english': 1,
    'spanish': 2,
    'french': 3,
    'german': 4,
    'hindi': 5,
    'japanese': 6,
    'chinese': 7,
    'italian': 8,
    'arabic': 9,
    'korean': 10,
    'bengali': 11
}

# Skill level mapping
SKILL_MAP = {
    'beginner': 1,
    'intermediate': 2,
    'advanced': 3
}

def encode_language(lang_name):
    """Convert language name (string) to numeric ID."""
    if isinstance(lang_name, str):
        lang_key = lang_name.strip().lower()
        return LANGUAGE_MAP.get(lang_key, 1)
    return int(lang_name)

def encode_skill(skill_name):
    """Convert skill name (string) to numeric ID."""
    if isinstance(skill_name, str):
        skill_key = skill_name.strip().lower()
        return SKILL_MAP.get(skill_key, 1)
    return int(skill_name)

def normalize_streak(streak, max_streak=100):
    """Normalize streak value to [0, 1] range."""
    return min(float(streak) / max_streak, 1.0)

def is_complementary_match(user_i, user_j):
    """Check if this is a complementary language exchange match."""
    known_i = encode_language(user_i.get('known_language', user_i.get('known_language_id', 1)))
    learn_i = encode_language(user_i.get('learning_language', user_i.get('learning_language_id', 1)))
    known_j = encode_language(user_j.get('known_language', user_j.get('known_language_id', 1)))
    learn_j = encode_language(user_j.get('learning_language', user_j.get('learning_language_id', 1)))
    
    return (learn_i == known_j and learn_j == known_i)

def is_peer_match(user_i, user_j):
    """Check if this is a peer learning match."""
    known_i = encode_language(user_i.get('known_language', user_i.get('known_language_id', 1)))
    learn_i = encode_language(user_i.get('learning_language', user_i.get('learning_language_id', 1)))
    known_j = encode_language(user_j.get('known_language', user_j.get('known_language_id', 1)))
    learn_j = encode_language(user_j.get('learning_language', user_j.get('learning_language_id', 1)))
    
    return (known_i == known_j and learn_i == learn_j)

def calculate_features(user_i, user_j, max_streak=100):
    """
    Calculates 12-feature vector for a pair of users with enhanced features.
    
    Returns: [A_known, A_learning, A_skill, A_streak, 
              B_known, B_learning, B_skill, B_streak,
              streak_diff, skill_diff, is_complementary, is_peer]
    """
    # Extract and encode user A features
    a_known = encode_language(user_i.get('known_language', user_i.get('known_language_id', 1)))
    a_learning = encode_language(user_i.get('learning_language', user_i.get('learning_language_id', 1)))
    a_skill = encode_skill(user_i.get('skill_level', 1))
    a_streak = normalize_streak(user_i.get('streak', user_i.get('learning_streak', 1)), max_streak)
    
    # Extract and encode user B features
    b_known = encode_language(user_j.get('known_language', user_j.get('known_language_id', 1)))
    b_learning = encode_language(user_j.get('learning_language', user_j.get('learning_language_id', 1)))
    b_skill = encode_skill(user_j.get('skill_level', 1))
    b_streak = normalize_streak(user_j.get('streak', user_j.get('learning_streak', 1)), max_streak)
    
    # NEW: Calculate derived features
    # Streak difference (normalized: 0 = same, 1 = max different)
    streak_diff = abs(a_streak - b_streak)
    
    # Skill level difference (normalized: 0 = same, 1 = max different)
    skill_diff = abs(a_skill - b_skill) / 2.0  # Max difference is 2 (1 to 3)
    
    # Match type flags
    is_comp = float(is_complementary_match(user_i, user_j))
    is_peer = float(is_peer_match(user_i, user_j))
    
    return [a_known, a_learning, a_skill, a_streak, 
            b_known, b_learning, b_skill, b_streak,
            streak_diff, skill_diff, is_comp, is_peer]

def generate_soft_labels(user_i, user_j):
    """
    Generate SOFT labels for better ranking instead of binary 0/1.
    
    Base scores:
    - Complementary match → 1.0 (perfect exchange)
    - Peer match → 0.85 (good for studying together)
    - No match → 0.0
    
    Adjustments:
    - Reduce by streak difference (smaller diff = better score)
    - Reduce by skill gap (smaller gap = better score)
    - Max reduction: 0.2 total
    """
    # Check match types
    is_comp = is_complementary_match(user_i, user_j)
    is_peer = is_peer_match(user_i, user_j)
    
    if not is_comp and not is_peer:
        return 0.0
    
    # Base score based on match type
    base_score = 1.0 if is_comp else 0.85
    
    # Get streak values
    a_streak = user_i.get('streak', user_i.get('learning_streak', 50))
    b_streak = user_j.get('streak', user_j.get('learning_streak', 50))
    streak_diff = abs(a_streak - b_streak) / 100.0  # Normalize to [0, 1]
    
    # Get skill levels
    a_skill = encode_skill(user_i.get('skill_level', 2))
    b_skill = encode_skill(user_j.get('skill_level', 2))
    skill_gap = abs(a_skill - b_skill) / 2.0  # Normalize to [0, 1]
    
    # Adjustment: penalize for differences
    # Large streak difference → reduce by ~0.1
    # Large skill gap → reduce by ~0.1
    adjustment = (streak_diff * 0.1) + (skill_gap * 0.1)
    
    final_score = max(0.0, base_score - adjustment)
    
    return float(final_score)

def generate_labels(user_i, user_j):
    """
    Backward compatible function.
    Returns 1.0 for any valid match, 0.0 otherwise (for classification).
    """
    is_comp = is_complementary_match(user_i, user_j)
    is_peer = is_peer_match(user_i, user_j)
    
    return 1.0 if (is_comp or is_peer) else 0.0

