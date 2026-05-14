"""Convert a curated CSV to the Stage 1 IRL expert-image dataset format.

Output layout:

    <output-dir>/<uid>/caption.txt
    <output-dir>/<uid>/<uid>.png
"""

import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing real images named <uid>.png. Only needed if CSV has no real_image_path column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_df = pd.read_csv(args.csv_file, dtype={"uid": str})
    print(f"uid number: {len(csv_df)}")

    for _, row in tqdm(csv_df.iterrows(), total=len(csv_df)):
        uid = row["uid"]
        prompt = row["prompt"]

        if "real_image_path" in csv_df.columns:
            real_image_path = row["real_image_path"]
        else:
            real_image_path = os.path.join(args.image_dir, f"{uid}.png")

        target_dir = os.path.join(args.output_dir, uid)
        os.makedirs(target_dir, exist_ok=True)

        target_image_path = os.path.join(target_dir, f"{uid}.png")
        shutil.copy(real_image_path, target_image_path)

        caption_path = os.path.join(target_dir, "caption.txt")
        with open(caption_path, "w") as f:
            f.write(prompt)


if __name__ == "__main__":
    main()
