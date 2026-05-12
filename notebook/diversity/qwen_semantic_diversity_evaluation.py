import os
import csv
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser(description="Classify images from a CSV file using Qwen3.5 VLM.")
parser.add_argument("--csv", required=True, help="Path to input CSV file with uid and win_image_path columns.")
parser.add_argument("--output", default=None, help="Path to output CSV file. Defaults to <csv_dir>/diversity_evaluation_results.csv.")
parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Model name or local path.")
parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens for generation.")
args = parser.parse_args()

model_name = args.model

processor = AutoProcessor.from_pretrained(model_name)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={
        0: "21GiB",
        1: "22GiB",
        2: "22GiB",
        3: "22GiB",
        "cpu": "64GiB",
    },
    low_cpu_mem_usage=True,
)

def get_input_device(model):
    for key in [
        "model.embed_tokens",
        "language_model.model.embed_tokens",
        "model.layers.0",
        "language_model.model.layers.0",
    ]:
        if key in model.hf_device_map:
            dev = model.hf_device_map[key]
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    for dev in model.hf_device_map.values():
        if isinstance(dev, int):
            return torch.device(f"cuda:{dev}")
        if isinstance(dev, str) and dev.startswith("cuda"):
            return torch.device(dev)
    return torch.device("cpu")

input_device = get_input_device(model)

prompt = """Please classify the given image into exactly one of the following 12 categories:

1. Characters (e.g., portraits, anime characters, fictional figures)
2. Arts (e.g., paintings, illustrations, abstract art)
3. Design (e.g., graphic design, UI design, posters)
4. Architecture (e.g., buildings, interiors, urban landscapes)
5. Animals
6. Natural Scenery (e.g., mountains, oceans, forests, skies)
7. Transportation (e.g., cars, aircraft, ships)
8. Products (e.g., consumer goods, industrial items)
9. Plants (e.g., flowers, trees, vegetation)
10. Food (e.g., dishes, beverages, ingredients)
11. Science (e.g., laboratories, scientific equipment, technology)
12. Others (does not fit any of the above categories)

Reply with only the category name in English (e.g., "Animals"). Do not include any explanation."""

csv_path = args.csv
output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(csv_path)), "diversity_evaluation_results.csv")

# Load existing results to support resume
processed_uids = set()
if os.path.exists(output_path):
    with open(output_path, "r", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_out)
        for row in reader:
            processed_uids.add(row["uid"])
    print(f"Resuming: {len(processed_uids)} already processed.")

with open(csv_path, "r", newline="", encoding="utf-8") as f_in, \
     open(output_path, "a", newline="", encoding="utf-8") as f_out:

    reader = csv.DictReader(f_in)
    fieldnames = ["uid", "win_image_path", "category"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)

    # Write header only if file is new
    if len(processed_uids) == 0:
        writer.writeheader()

    rows = list(reader)
    total = len(rows)
    pending = [r for r in rows if r["uid"] not in processed_uids]

    pbar = tqdm(total=len(pending), desc="Classifying", unit="img", dynamic_ncols=True)

    for i, row in enumerate(rows):
        uid = row["uid"]
        image_path = row["win_image_path"]

        if uid in processed_uids:
            continue

        if not os.path.exists(image_path):
            pbar.set_postfix_str(f"SKIP {uid[:8]}")
            writer.writerow({"uid": uid, "win_image_path": image_path, "category": "N/A"})
            f_out.flush()
            pbar.update(1)
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        category = output_text[0].strip()
        pbar.set_postfix_str(f"{uid[:8]} → {category}")
        writer.writerow({"uid": uid, "win_image_path": image_path, "category": category})
        f_out.flush()
        pbar.update(1)

    pbar.close()

print(f"\nDone. Results saved to: {output_path}")
