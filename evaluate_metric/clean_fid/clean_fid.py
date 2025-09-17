import os
import argparse
from cleanfid import fid

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--metric", type=str, 
)
parser.add_argument(
    "--reference_image_dir", type=str, 
)
parser.add_argument(
    "--generated_image_dir", type=str,
)
args = parser.parse_args()

if args.metric == "Clean-FID":
    fid_score = fid.compute_fid(args.reference_image_dir, args.generated_image_dir)
    print(f"Clean-FID Score: {fid_score}")

elif args.metric == "Clean-KID":
    kid_score = fid.compute_kid(args.reference_image_dir, args.generated_image_dir)
    print(f"Clean-KID Score: {kid_score}")

elif args.metric == "CLIP-FID":
    clip_fid_score = fid.compute_fid(args.reference_image_dir, args.generated_image_dir, mode="clean", model_name="clip_vit_b_32")
    print(f"CLIP-FID Score: {clip_fid_score}")