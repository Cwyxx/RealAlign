import os
import csv
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# # Configuration
# image_dir = "/data3/xy/proj/bench/dataset/train/real"
# uid_file = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/qwen_3_caption/uid.csv"
# ext_list = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
# output_file = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/qwen_3_caption/qwen3_caption_results.csv"  # Output file path

# Configuration
image_dir = "/data_center/data2/dataset/chenwy/21164-data/genimage/stable_diffusion_v_1_5/train/nature"
uid_file = None
ext_list = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
output_file = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/imagenet/imagenet_qwen3_caption_results.csv"  # Output file path

# Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Read UIDs from CSV file
def read_uids_from_csv(csv_path):
    """Read UIDs from CSV file (first column after header)"""
    uids = []
    if csv_path is None:
        return uids
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and row[0].strip():  # Check if row is not empty
                uids.append(row[0].strip())
    return uids

# Find image file for a given UID
def find_image_file(image_dir, uid, ext_list):
    """Find image file with given UID and any of the extensions"""
    image_dir_path = Path(image_dir)
    for ext in ext_list:
        image_path = image_dir_path / f"{uid}{ext}"
        if image_path.exists():
            return str(image_path)
    return None

# Generate caption for a single image
def generate_caption(image_path):
    """Generate caption for an image using Qwen3-VL"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Describe this image in a single sentence. Only output the caption, no other text."},
            ],
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""

# Read already processed results from output file
def read_processed_results(output_path):
    """Read already processed results from CSV file"""
    processed_uids = set()
    existing_results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = row.get("uid", "").strip()
                    if uid:
                        processed_uids.add(uid)
                        existing_results.append({
                            "uid": uid,
                            "image_path": row.get("image_path", ""),
                            "prompt": row.get("prompt", ""),
                            "error": row.get("error", "")
                        })
        except Exception as e:
            print(f"Warning: Could not read existing results from {output_path}: {e}")
    return processed_uids, existing_results

# Save results to file
def save_results(results, output_path):
    """Save results to CSV file"""
    fieldnames = ["uid", "image_path", "prompt", "error"]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "uid": result.get("uid", ""),
                "image_path": result.get("image_path", ""),
                "prompt": result.get("prompt", ""),
                "error": result.get("error", "")
            }
            writer.writerow(row)
    print(f"  Results saved to {output_path}")

# Main processing
if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Read already processed results
    print(f"Checking existing results in {output_file}...")
    processed_uids, existing_results = read_processed_results(output_file)
    print(f"Found {len(processed_uids)} already processed UIDs")
    
    # Read UIDs
    if uid_file is not None:
        print(f"Reading UIDs from {uid_file}...")
        all_uids = read_uids_from_csv(uid_file)
        print(f"Total UIDs in input file: {len(all_uids)}")
    else:
        all_uids = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]
        print(f"Total UIDs in input directory: {len(all_uids)}")
    
    # Filter out already processed UIDs
    uids_to_process = [uid for uid in all_uids if uid not in processed_uids]
    print(f"UIDs to process: {len(uids_to_process)} (skipping {len(all_uids) - len(uids_to_process)} already processed)")
    
    if len(uids_to_process) == 0:
        print("All UIDs have already been processed. Exiting.")
        exit(0)
    
    # Start with existing results
    results = existing_results.copy()
    
    # Process each UID
    for idx, uid in enumerate(uids_to_process, 1):
        print(f"[{idx}/{len(uids_to_process)}] Processing UID: {uid}")
        
        # Find image file
        image_path = find_image_file(image_dir, uid, ext_list)
        if image_path is None:
            print(f"  Warning: Image not found for UID {uid}")
            result = {"uid": uid, "image_path": None, "prompt": None}
            results.append(result)
            # Save results immediately
            save_results(results, output_file)
            continue
        
        print(f"  Found image: {image_path}")
        
        # Generate caption
        try:
            prompt = generate_caption(image_path)
            print(f"  Prompt: {prompt}")
            result = {"uid": uid, "image_path": image_path, "prompt": prompt}
            results.append(result)
            # Save results immediately after getting a caption
            save_results(results, output_file)
        except Exception as e:
            print(f"  Error processing {uid}: {e}")
            result = {"uid": uid, "image_path": image_path, "prompt": None, "error": str(e)}
            results.append(result)
            # Save results even if there's an error
            save_results(results, output_file)
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Summary:")
    print(f"Total UIDs in input: {len(all_uids)}")
    print(f"Already processed: {len(processed_uids)}")
    print(f"Processed in this run: {len(uids_to_process)}")
    print(f"Successfully processed (this run): {sum(1 for r in results[len(existing_results):] if r.get('prompt') is not None)}")
    print(f"Failed/Not found (this run): {sum(1 for r in results[len(existing_results):] if r.get('prompt') is None)}")
    print(f"Total results: {len(results)}")
    print(f"Total with prompts: {sum(1 for r in results if r.get('prompt') is not None)}")
    
    # Print all results
    print("\nResults:")
    for result in results:
        print(f"UID: {result['uid']}")
        if result.get('prompt'):
            print(f"  Prompt: {result['prompt']}")
        else:
            print(f"  Status: Failed or not found")
