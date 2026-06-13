import sys
import os
from data_loader import generate_synthetic_csv
from train import run_training
from recommend import recommend_partners, display_recommendations

# Configure UTF-8 encoding for stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr:
            sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def main():
    print("\n" + "="*80)
    print(f"{'DEEP LEARNING LANGUAGE PARTNER RECOMMENDER - AUTOMATED RUN':^80}")
    print("="*80 + "\n")
    
    # 1. Data Generation
    print("[1/3] Generating Synthetic Data...")
    generate_synthetic_csv()
    
    # 2. Training
    print("\n[2/3] Training Deep Neural Network...")
    run_training(epochs=20)
    
    # 3. Recommendation Generation
    print("\n[3/3] Generating Top 10 Recommendations for User #1...")
    recs = recommend_partners(1, top_n=10)
    display_recommendations(1, recs)
    
    print("\n[4/4] Generating Top 10 Recommendations for User #2...")
    recs2 = recommend_partners(2, top_n=10)
    display_recommendations(2, recs2)
    
    print("\nAutomated run completed. Thank you!")

if __name__ == "__main__":
    main()
