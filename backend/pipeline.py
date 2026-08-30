import os
import pandas as pd
from attack_generator import run_attack_generator
from feedback_loop import run_feedback_loop

def main():
    print("INIT: RED vs BLUE TEAM AUTOMATED PIPELINE")
    
    unnoticed_frauds_path = os.path.join(os.path.dirname(__file__), 'attacks', 'unnoticed_frauds.csv')
    
    unnoticed_frauds_df = None
    if os.path.exists(unnoticed_frauds_path):
        print(f"\n[PIPELINE] Found highly-evasive frauds from previous run: {unnoticed_frauds_path}")
        unnoticed_frauds_df = pd.read_csv(unnoticed_frauds_path)
        print(f"[PIPELINE] Loading {len(unnoticed_frauds_df)} unnoticed frauds to bolster next attack.")
    else:
        print("\n[PIPELINE] No previous unnoticed frauds found. Generating purely novel attack.")
    
    print("\nSTAGE 1: RED TEAM ATTACK GENERATION")
    attack_file = run_attack_generator(unnoticed_frauds_df=unnoticed_frauds_df)
    
    if not attack_file:
        print("[PIPELINE] Error generating attack. Pipeline aborted.")
        return
        
    print("\nSTAGE 2: BLUE TEAM DEFENSE & FINE-TUNING")
    new_unnoticed_df = run_feedback_loop(attack_file)
    
    print("PIPELINE CYCLE COMPLETE")
    print(f"Next cycle will use {len(new_unnoticed_df)} highly-evasive frauds for an even stronger attack.")

if __name__ == "__main__":
    main()
