import torch
import pandas as pd
import numpy as np
from data_loader import load_data
from feature_engineering import calculate_features, LANGUAGE_MAP
from model import CompatibilityModel
import os
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

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    print("\n" + Colors.BOLD + Colors.CYAN + "="*80 + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + f"{text:^80}" + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + "="*80 + Colors.ENDC + "\n")

def print_section(text):
    """Print a section header"""
    print(Colors.BOLD + Colors.BLUE + f"\n📌 {text}" + Colors.ENDC)
    print(Colors.BLUE + "-" * 80 + Colors.ENDC)

def get_valid_language():
    """Get valid language input from user"""
    while True:
        lang = input(Colors.YELLOW + "Enter language name: " + Colors.ENDC).strip().lower()
        if lang in LANGUAGE_MAP:
            return lang.capitalize()
        else:
            available = ", ".join([l.capitalize() for l in LANGUAGE_MAP.keys()])
            print(Colors.RED + f"❌ Invalid language. Available: {available}" + Colors.ENDC)

def get_user_profile():
    """Get user profile information"""
    print_section("USER PROFILE SETUP")
    
    print("Available languages:")
    for i, lang in enumerate(sorted(LANGUAGE_MAP.keys()), 1):
        print(f"  {i:2d}. {lang.capitalize()}")
    
    print(Colors.YELLOW + "\n→ Enter your NATIVE language (the language you speak at home):" + Colors.ENDC)
    known_lang = get_valid_language()
    
    print(Colors.YELLOW + "\n→ Enter the language you want to LEARN:" + Colors.ENDC)
    learning_lang = get_valid_language()
    
    if known_lang == learning_lang:
        print(Colors.RED + "❌ Native and learning languages must be different!" + Colors.ENDC)
        return get_user_profile()
    
    return known_lang, learning_lang

def load_model():
    """Load trained model"""
    model = CompatibilityModel()
    try:
        model.load_state_dict(torch.load('compatibility_model.pt', weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        print(Colors.RED + "❌ Model not found! Please run: python main.py" + Colors.ENDC)
        return None

def find_recommendations(known_lang, learning_lang, top_n=10):
    """
    Find recommendations using RULE-BASED FILTERING + AI RANKING.
    
    Step 1: FILTER - Only users with valid language compatibility
    (Complementary Exchange OR Peer Learning)
    
    Step 2: RANK - Use AI model to score filtered users
    with consideration for streak and skill similarity
    """
    from feature_engineering import is_complementary_match, is_peer_match
    
    df_users = load_data()
    model = load_model()
    
    if model is None:
        return None, None
    
    # Create target user profile
    target_user = {
        'user_id': -1,
        'known_language': known_lang,
        'learning_language': learning_lang,
        'skill_level': 2,
        'streak': 50
    }
    
    print(Colors.CYAN + f"\n🔍 Searching database of {len(df_users)} users..." + Colors.ENDC)
    
    # STEP 1: FILTER - Only keep valid matches
    print(Colors.CYAN + "  • Filtering for valid language compatibility..." + Colors.ENDC)
    valid_users = []
    valid_indices = []
    
    for idx, user in df_users.iterrows():
        user_dict = user.to_dict()
        # Keep only complementary or peer matches
        if is_complementary_match(target_user, user_dict) or is_peer_match(target_user, user_dict):
            valid_users.append(user_dict)
            valid_indices.append(idx)
    
    if len(valid_users) == 0:
        print(Colors.RED + "❌ No compatible language partners found!" + Colors.ENDC)
        return None, None
    
    print(Colors.CYAN + f"  • Found {len(valid_users)} valid matches" + Colors.ENDC)
    
    # STEP 2: RANK - Use AI model to score filtered users
    print(Colors.CYAN + "  • Running AI model to rank matches by compatibility..." + Colors.ENDC)
    features_list = []
    
    for user_dict in valid_users:
        features = calculate_features(target_user, user_dict)
        features_list.append(features)
    
    # Get AI predictions (scores between 0 and 1)
    X_tensor = torch.tensor(features_list, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_tensor).numpy().flatten()
    
    # Add scores to the filtered dataframe
    df_valid = df_users.iloc[valid_indices].copy()
    df_valid['predicted_score'] = scores
    
    # Determine match type
    def get_match_type(user_row):
        if (target_user['learning_language'] == user_row['known_language'] and 
            user_row['learning_language'] == target_user['known_language']):
            return "🔄 COMPLEMENTARY EXCHANGE"
        elif (target_user['known_language'] == user_row['known_language'] and 
              target_user['learning_language'] == user_row['learning_language']):
            return "👥 PEER LEARNING"
        return "📚 SIMILAR INTERESTS"
    
    df_valid['match_type'] = df_valid.apply(get_match_type, axis=1)
    
    # Sort by score (AI ranking)
    recommendations = df_valid.sort_values(by='predicted_score', ascending=False).head(top_n)
    
    print(Colors.GREEN + "  ✓ Ranking complete!" + Colors.ENDC)
    
    return recommendations, target_user

def display_recommendations(recommendations, target_user, top_n=10):
    """Display detailed recommendations"""
    if recommendations is None or len(recommendations) == 0:
        print(Colors.RED + "❌ No recommendations found!" + Colors.ENDC)
        return
    
    print_section("🌟 TOP LANGUAGE LEARNING PARTNERS")
    
    print(Colors.GREEN + f"YOUR PROFILE:" + Colors.ENDC)
    print(f"  • Native Language: {Colors.BOLD}{target_user['known_language']}{Colors.ENDC}")
    print(f"  • Target Language: {Colors.BOLD}{target_user['learning_language']}{Colors.ENDC}")
    print()
    
    print(Colors.GREEN + f"MATCH TYPES EXPLAINED:" + Colors.ENDC)
    print(f"  🔄 {Colors.BOLD}COMPLEMENTARY EXCHANGE{Colors.ENDC}")
    print(f"     → They speak what YOU want to learn")
    print(f"     → They want to learn what YOU speak")
    print(f"     → Perfect for mutual language exchange!")
    print()
    print(f"  👥 {Colors.BOLD}PEER LEARNING{Colors.ENDC}")
    print(f"     → Same native language as you")
    print(f"     → Learning the same target language")
    print(f"     → Great for studying together!")
    print()
    
    print(Colors.CYAN + "="*100 + Colors.ENDC)
    print(f"{Colors.BOLD}{'RANK':<6} {'USER ID':<10} {'COMPATIBILITY':<6} {'MATCH TYPE':<25} {'NATIVE':<15} {'LEARNING':<15} {'STREAK':<8}{Colors.ENDC}")
    print(Colors.CYAN + "="*100 + Colors.ENDC)
    
    for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
        score_color = Colors.GREEN if row['predicted_score'] >= 0.8 else Colors.YELLOW
        
        print(f"#{rank:<5} {int(row['user_id']):<10} {score_color}{row['predicted_score']:.1%}{Colors.ENDC}       "
              f"{row['match_type']:<25} {row['known_language']:<15} {row['learning_language']:<15} {int(row['streak']) if not pd.isna(row['streak']) else 'N/A':<8}")
    
    print(Colors.CYAN + "="*100 + Colors.ENDC)
    
    # Print detailed information for top 3
    print()
    print_section("📊 DETAILED PROFILES (TOP 3 MATCHES)")
    
    for rank, (_, row) in enumerate(recommendations.head(3).iterrows(), 1):
        score_pct = f"{row['predicted_score']:.1%}"
        color = Colors.GREEN if row['predicted_score'] >= 0.8 else Colors.YELLOW
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}#{rank} - USER #{int(row['user_id'])}{Colors.ENDC}")
        print(Colors.BLUE + "-" * 50 + Colors.ENDC)
        print(f"  {Colors.BOLD}Match Score:{Colors.ENDC} {color}{score_pct}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Match Type:{Colors.ENDC} {row['match_type']}")
        print(f"  {Colors.BOLD}Native Language:{Colors.ENDC} {row['known_language']}")
        print(f"  {Colors.BOLD}Learning Language:{Colors.ENDC} {row['learning_language']}")
        print(f"  {Colors.BOLD}Learning Streak:{Colors.ENDC} {int(row['streak']) if not pd.isna(row['streak']) else 'N/A'} days")
        print(f"  {Colors.BOLD}Skill Level:{Colors.ENDC} {int(row['skill_level']) if not pd.isna(row['skill_level']) else 'N/A'}/3")

def show_statistics(recommendations):
    """Show statistics about recommendations"""
    if recommendations is None or len(recommendations) == 0:
        return
    
    print_section("📈 STATISTICS")
    
    print(f"  • Total Matches Found: {len(recommendations)}")
    print(f"  • Average Compatibility Score: {recommendations['predicted_score'].mean():.1%}")
    print(f"  • Highest Score: {recommendations['predicted_score'].max():.1%}")
    print(f"  • Lowest Score: {recommendations['predicted_score'].min():.1%}")
    
    match_types = recommendations['match_type'].value_counts()
    print(f"\n  • Match Type Distribution:")
    for match_type, count in match_types.items():
        print(f"    - {match_type}: {count} matches")

def main():
    """Main interactive application"""
    print_header("🌍 LANGUAGE LEARNING PARTNER RECOMMENDER 🌍")
    print(Colors.CYAN + "Find your perfect language exchange partner using AI!" + Colors.ENDC)
    
    while True:
        # Get user profile
        known_lang, learning_lang = get_user_profile()
        
        # Find recommendations
        recommendations, target_user = find_recommendations(known_lang, learning_lang, top_n=10)
        
        # Display results
        if recommendations is not None:
            display_recommendations(recommendations, target_user, top_n=10)
            show_statistics(recommendations)
        
        # Ask if user wants to continue
        print()
        print(Colors.YELLOW + "Would you like to find partners for different languages? (yes/no): " + Colors.ENDC, end="")
        choice = input().strip().lower()
        
        if choice not in ['yes', 'y']:
            print_header("Thank you for using Language Partner Recommender! 👋")
            print(Colors.GREEN + "Good luck with your language learning journey! 🚀" + Colors.ENDC)
            break

if __name__ == "__main__":
    main()
