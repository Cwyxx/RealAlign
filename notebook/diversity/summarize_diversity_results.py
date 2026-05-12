import os
import csv
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Summarize category distribution from diversity evaluation results.")
parser.add_argument("--input", required=True, help="Path to diversity_evaluation_results.csv")
parser.add_argument("--output", default=None, help="Path to output summary CSV. Defaults to <input_dir>/diversity_summary.csv.")
args = parser.parse_args()

output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.input)), "diversity_summary.csv")

counter = Counter()
total = 0

with open(args.input, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        category = row["category"].strip()
        counter[category] += 1
        total += 1

print(f"\nTotal images: {total}")
print(f"{'Category':<30} {'Count':>6}  {'Percentage':>10}")
print("-" * 50)

rows = sorted(counter.items(), key=lambda x: x[1], reverse=True)
for category, count in rows:
    pct = count / total * 100 if total > 0 else 0
    print(f"{category:<30} {count:>6}  {pct:>9.2f}%")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "count", "percentage"])
    writer.writeheader()
    for category, count in rows:
        pct = count / total * 100 if total > 0 else 0
        writer.writerow({"category": category, "count": count, "percentage": f"{pct:.2f}%"})

print(f"\nSummary saved to: {output_path}")
