import pandas as pd
import torch
import numpy as np
from recommendation_engine import LanguagePartnerModel, recommend_partners
from preprocess_data import load_and_preprocess_data

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load necessary data
df_users = pd.read_csv('users.csv')
_, _, _, _, scaler = load_and_preprocess_data('synthetic_pairs.csv')

# Load model
model = LanguagePartnerModel()
model.load_state_dict(torch.load('language_partner_model.pt', weights_only=True))
model.eval()

# Pick target user
target_user = df_users.iloc[0].to_dict()

# Get recommendations
recommendations = recommend_partners(target_user, df_users, model, scaler, top_n=10)

# Print cleanly
print("="*60)
print("Language Partner Recommendation Results")
print("="*60)
print(f"Target User: ID {target_user['user_id']}")
print(f"Speaks: Lang {target_user['known_language_id']}, Learning: Lang {target_user['learning_language_id']}")
print(f"Skill Level: {target_user['skill_level_id']}, Streak: {target_user['learning_streak']} days")
print("-" * 60)
print("Top 10 Recommended Partners:")
print("-" * 60)
header = f"{'Partner ID':<12} | {'Speaks':<8} | {'Learning':<8} | {'Skill':<6} | {'Compatibility'}"
print(header)
print("-" * len(header))
for _, row in recommendations.iterrows():
    print(f"{int(row['user_id']):<12} | {int(row['known_language_id']):<8} | {int(row['learning_language_id']):<8} | {int(row['skill_level_id']):<6} | {row['compatibility_score']:.4f}")
print("="*60)
