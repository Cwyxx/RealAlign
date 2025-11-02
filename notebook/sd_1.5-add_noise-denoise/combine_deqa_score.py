#!/usr/bin/env python3
"""
combine deqa_fake_score.csv and deqa_real_score.csv into deqa_score.csv
"""
import pandas as pd
import os

base_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/deqa"
fake_file = os.path.join(base_dir, "deqa_fake_score.csv")
real_file = os.path.join(base_dir, "deqa_real_score.csv")
output_file = os.path.join(base_dir, "deqa_score.csv")

print(f"Reading {fake_file}...")
df_fake = pd.read_csv(fake_file)
print(f"Fake score file rows: {len(df_fake)}")

print(f"Reading {real_file}...")
df_real = pd.read_csv(real_file)
print(f"Real score file rows: {len(df_real)}")

# Rename deqa_score column to deqa_fake_score and deqa_real_score
df_fake = df_fake.rename(columns={'deqa_score': 'fake_image_score'})
df_real = df_real.rename(columns={'deqa_score': 'real_image_score'})

df_fake_subset = df_fake[['uid', 'fake_image_score']]
df_real_subset = df_real[['uid', 'real_image_score']]

print("Merging data...")
df_merged = pd.merge(df_fake_subset, df_real_subset, on='uid', how='inner')

print(f"Merged data rows: {len(df_merged)}")

# Check if there are missing uids
if len(df_merged) != len(df_fake):
    print(f"Warning: Merged rows ({len(df_merged)}) do not match original file rows ({len(df_fake)})")
    # Check which uids are missing
    fake_uids = set(df_fake['uid'])
    real_uids = set(df_real['uid'])
    missing_in_real = fake_uids - real_uids
    missing_in_fake = real_uids - fake_uids
    if missing_in_real:
        print(f"Number of uids missing in real file: {len(missing_in_real)}")
    if missing_in_fake:
        print(f"Number of uids missing in fake file: {len(missing_in_fake)}")

# Save merged file
print(f"Saving to {output_file}...")
df_merged.to_csv(output_file, index=False)
print(f"Done! Merged file saved to: {output_file}")

# Show first few rows as preview
print("\n前5行预览:")
print(df_merged.head())

