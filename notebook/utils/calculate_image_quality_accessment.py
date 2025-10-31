# simple_score_calculation.py
import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM
import pyiqa
import argparse
import time

def calculate_deqa_scores(model, image_paths, batch_size=8):
    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Calculating DeQA scores"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(path).convert('RGB') for path in batch_paths]
        batch_scores = model.score(batch_images).tolist()
        scores.extend(batch_scores)
    return scores

def calculate_qalign_scores(model, image_paths):
    scores = []
    for image_path in tqdm(image_paths, desc="Calculating Q-Align scores"):
        score = model(image_path)
        if isinstance(score, torch.Tensor):
            score = score.item()
        scores.append(score)
    return scores

def main():
    parser = argparse.ArgumentParser(description='Calculate image quality scores')
    parser.add_argument(
        '--metric',
        type=str,
        choices=['deqa', 'qalign', 'both'],
        default='both',
        help='Which metric to calculate: deqa, qalign, or both (default: both)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train.csv",
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/fake",
        help='Base directory for source images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step",
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for DeQA calculation (default: 8)'
    )
    
    args = parser.parse_args()
    
    val_csv_file_path = args.csv_path
    base_source_image_dir = args.image_dir
    output_dir = args.output_dir
    output_csv_path = os.path.join(output_dir, f"{args.metric}_fake_score.csv")
    
    metric_mode = args.metric
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Configuration:")
    print(f"  Metric mode: {metric_mode}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input CSV: {val_csv_file_path}")
    print(f"  Image dir: {base_source_image_dir}")
    print(f"  Output dir: {output_dir}")
    print("="*60 + "\n")
    
    print("Loading and filtering dataset...")
    df = pd.read_csv(val_csv_file_path)
    # df = df.head(10000)
    print(f"df length: {len(df)}")
    
    print("\nChecking image existence...")
    ext_list = [".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        uid = row['uid']
        found = False
        
        for ext in ext_list:
            potential_file = os.path.join(base_source_image_dir, f"{uid}{ext}")
            if os.path.exists(potential_file):
                valid_data.append({
                    'uid': uid,
                    'image_path_full': potential_file,
                })
                found = True
                break
        
        if not found:
            print(f"Warning: Image not found for uid {uid}")
    
    print(f"Valid images found: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("No valid images found. Exiting...")
        return
    
    uids = [item['uid'] for item in valid_data]
    image_paths = [item['image_path_full'] for item in valid_data]
    
    start_time = time.time()
    
    results_dict = {}
    
    if metric_mode in ['deqa', 'both']:
        print("\n" + "="*60)
        print("Initializing DeQA model...")
        print("="*60)
        deqa_model = AutoModelForCausalLM.from_pretrained(
            "zhiyuanyou/DeQA-Score-Mix3",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        print("DeQA model loaded successfully!\n")
        
        print("Calculating DeQA scores...")
        deqa_scores = calculate_deqa_scores(deqa_model, image_paths, args.batch_size)
        results_dict['deqa'] = deqa_scores
        
        del deqa_model
        torch.cuda.empty_cache()
        print("DeQA calculation completed!\n")
    
    if metric_mode in ['qalign', 'both']:
        print("="*60)
        print("Initializing Q-Align model...")
        print("="*60)
        qalign_model = pyiqa.create_metric("qalign", device=device)
        print("Q-Align model loaded successfully!\n")
        
        print("Calculating Q-Align scores...")
        qalign_scores = calculate_qalign_scores(qalign_model, image_paths)
        results_dict['qalign'] = qalign_scores
        
        del qalign_model
        torch.cuda.empty_cache()
        print("Q-Align calculation completed!\n")
    
    end_time = time.time()
    print(f"Total calculation time: {end_time - start_time:.2f} seconds\n")
    
    print("Organizing results...")
    
    results_df = pd.DataFrame({
        'uid': uids,
        'image_path_full': image_paths,
    })
    
    if 'deqa' in results_dict:
        results_df['deqa_score'] = results_dict['deqa']
    
    if 'qalign' in results_dict:
        results_df['qalign_score'] = results_dict['qalign']
    
    print("\nSaving results to CSV...")
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"✓ Results saved to: {output_csv_path}")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {list(results_df.columns)}")
    
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(f"Total images processed: {len(results_df)}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if 'deqa' in results_dict:
        print(f"\nDeQA scores:")
        print(f"  Mean: {results_df['deqa_score'].mean():.4f}")
        print(f"  Std:  {results_df['deqa_score'].std():.4f}")
        print(f"  Min:  {results_df['deqa_score'].min():.4f}")
        print(f"  Max:  {results_df['deqa_score'].max():.4f}")
    
    if 'qalign' in results_dict:
        print(f"\nQ-Align scores:")
        print(f"  Mean: {results_df['qalign_score'].mean():.4f}")
        print(f"  Std:  {results_df['qalign_score'].std():.4f}")
        print(f"  Min:  {results_df['qalign_score'].min():.4f}")
        print(f"  Max:  {results_df['qalign_score'].max():.4f}")
    
    print("="*50)
    print("\nSample results (first 5 rows):")
    print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
