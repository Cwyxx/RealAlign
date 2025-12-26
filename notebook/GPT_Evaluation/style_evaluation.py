import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import regex as re
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image
import argparse
import json
import base64
from openai import OpenAI

# ================= Configuration =================
parser = argparse.ArgumentParser(description="Classify image style (Realistic vs CG Render) using LLM.")
# 只需要一个图片目录
parser.add_argument('--image_dir', type=str, required=True, help="Directory containing the images to classify.")
# txt_path 可选，如果文件名就是 prompt 或者你只需要根据文件名索引，可以不依赖它。
# 这里保留它是为了方便你对应 UID，如果不需要 prompt 内容参与判断，可以不用读文件内容
parser.add_argument("--txt_path", type=str, default=None, help="Optional: Path to json/csv containing prompts if needed for metadata.")
parser.add_argument('--model', type=str, default="gpt-4o", help="Name of the large language model to use.")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the results.")
args = parser.parse_args()

# [安全提示] 请使用环境变量或在此处填入新的 Key，不要直接上传含 Key 的代码
private_api_key = "sk-SpCI4PAYOSqh8uQ8qf42XGTm8CttGwKGwyrWSdmnZl4GmObD"

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=private_api_key
)

# ================= Helper Functions =================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_json_response(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(content[start:end])
        except:
            pass
    return None

def load_prompts_from_txt(txt_path):
    if not txt_path or not os.path.exists(txt_path):
        return {}
    prompts = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()    
        uid = f"{i:05d}" 
        prompts[uid] = line
    return prompts

# ================= Main Logic =================

# 1. Scan Images
print(f"Scanning images in {args.image_dir}...")
exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
image_map = {} # uid -> filename

for f in os.listdir(args.image_dir):
    ext = os.path.splitext(f)[-1]
    if ext in exts:
        # 假设文件名就是 UID，或者你可以根据需要修改这部分逻辑
        uid = os.path.splitext(f)[0]
        image_map[uid] = f

uids = sorted(list(image_map.keys()))
print(f"Found {len(uids)} images.")

# Load prompts strictly for metadata (optional)
prompt_dict = load_prompts_from_txt(args.txt_path)

# Deduplicate prompts: keep only the first uid for each unique prompt
seen_prompts = set()
unique_uids = []
prompt_to_first_uid = {}  # Track which uid was kept for each prompt

for uid in uids:
    prompt = prompt_dict.get(uid, "")
    # If prompt is empty or already seen, skip this uid
    if prompt and prompt in seen_prompts:
        print(f"Skipping {uid}: duplicate prompt (first seen at {prompt_to_first_uid[prompt]})")
        continue
    # If prompt is empty, still process (no prompt-based deduplication)
    if prompt:
        seen_prompts.add(prompt)
        prompt_to_first_uid[prompt] = uid
    unique_uids.append(uid)

print(f"After deduplication: {len(unique_uids)} unique images (removed {len(uids) - len(unique_uids)} duplicates)")

# 2. Prepare Output
os.makedirs(args.output_dir, exist_ok=True)
output_csv_path = os.path.join(args.output_dir, f"{args.model}_style_classification.csv")
results = []

# 3. Prompts
system_prompt = "You are an expert computer vision assistant specialized in image style analysis."

# 核心 Prompt 修改：只针对单张图片提问
user_prompt_template = """Is the image in a realistic style?
Provide your response in strictly valid JSON format with two keys:
1. "is_realistic": A string, either "Yes" or "No".
2. "reason": A concise explanation (1 sentence).
"""

# 4. Loop
for uid in tqdm(unique_uids, desc=f'Classifying {args.model}'):
    # Check resume
    # if os.path.exists(output_csv_path) and uid in pd.read_csv(output_csv_path)['uid'].values: continue

    img_name = image_map[uid]
    image_path = os.path.join(args.image_dir, img_name)
    
    # Encode
    b64_img = encode_image(image_path)
    
    # Prompt string (unchanged per image, but necessary structure)
    formatted_prompt = user_prompt_template
    
    time.sleep(1) # Rate limit handling
    
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    ],
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        parsed_data = parse_json_response(raw_content)
        
        # Get optional original text prompt for reference
        original_prompt = prompt_dict.get(uid, "")
        
        if parsed_data:
            is_realistic = parsed_data.get("is_realistic") # 结果是 "Yes" 或 "No"
            reason = parsed_data.get("reason")
            
            results.append({
                "uid": uid,
                "original_prompt": original_prompt,
                "is_realistic": is_realistic, # CSV 表头也建议改成这个
                "reason": reason,
                "filename": img_name
            })
            
            # 打印输出也对应改一下
            print("-" * 30)
            print(f"UID: {uid} | Realistic: {is_realistic}")
            print(f"Reason: {reason}")
            
        else:
            print(f"\n[Error] JSON Parse Error for {uid}: {raw_content}")
            results.append({
                "uid": uid,
                "original_prompt": original_prompt,
                "classification": "Error",
                "reason": "JSON Parse Error",
                "filename": img_name
            })

    except Exception as e:
        print(f"\n[Exception] Error processing {uid}: {e}")
        time.sleep(2)

    # Periodic Save
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv(output_csv_path, index=False)

# Final Save
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"Done. Results saved to {output_csv_path}")