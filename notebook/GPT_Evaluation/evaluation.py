import os
import pandas as pd
import argparse
from collections import Counter

# ================= Configuration =================
parser = argparse.ArgumentParser(description="Evaluate comparison results and calculate win rates.")
parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing comparison results.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory containing the results CSV (alternative to --csv_path).")
parser.add_argument("--model", type=str, default=None, help="Model name (used with --output_dir to find CSV file).")
args = parser.parse_args()

# ================= Helper Functions =================
def calculate_win_rates(df):
    """Calculate win rates for Image 1 vs Image 2"""
    total = len(df)
    
    # Count selections
    selection_counts = Counter(df['selection'].astype(int))
    
    image1_wins = selection_counts.get(1, 0)  # Image 1 selected
    image2_wins = selection_counts.get(2, 0)  # Image 2 selected
    errors = selection_counts.get(-1, 0)      # Parse errors
    
    valid_comparisons = image1_wins + image2_wins
    
    # Calculate win rates
    if valid_comparisons > 0:
        image1_win_rate = image1_wins / valid_comparisons * 100
        image2_win_rate = image2_wins / valid_comparisons * 100
    else:
        image1_win_rate = 0
        image2_win_rate = 0
    
    return {
        'total': total,
        'image1_wins': image1_wins,
        'image2_wins': image2_wins,
        'errors': errors,
        'valid_comparisons': valid_comparisons,
        'image1_win_rate': image1_win_rate,
        'image2_win_rate': image2_win_rate
    }

# ================= Main Logic =================

# Determine CSV path
if args.csv_path:
    csv_path = args.csv_path
elif args.output_dir and args.model:
    csv_path = os.path.join(args.output_dir, f"{args.model}_comparison_results.csv")
else:
    raise ValueError("Either --csv_path or both --output_dir and --model must be provided.")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

print(f"Loading results from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"\n{'='*60}")
print(f"Evaluation Results")
print(f"{'='*60}")

# Calculate statistics
stats = calculate_win_rates(df)

print(f"\nTotal comparisons: {stats['total']}")
print(f"Valid comparisons: {stats['valid_comparisons']}")
print(f"Parse errors: {stats['errors']}")

print(f"\n{'─'*60}")
print(f"Win Rate Statistics:")
print(f"{'─'*60}")
print(f"Image 1 wins: {stats['image1_wins']} ({stats['image1_win_rate']:.2f}%)")
print(f"Image 2 wins: {stats['image2_wins']} ({stats['image2_win_rate']:.2f}%)")

print(f"\n{'─'*60}")
print(f"Summary:")
print(f"{'─'*60}")
if stats['image1_win_rate'] > stats['image2_win_rate']:
    print(f"✓ Image 1 is preferred (win rate: {stats['image1_win_rate']:.2f}%)")
elif stats['image2_win_rate'] > stats['image1_win_rate']:
    print(f"✓ Image 2 is preferred (win rate: {stats['image2_win_rate']:.2f}%)")
else:
    print(f"= Tie (both have {stats['image1_win_rate']:.2f}% win rate)")

print(f"\n{'='*60}")

