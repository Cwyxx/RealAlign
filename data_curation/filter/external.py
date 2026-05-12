"""Filter external-source (real, fake) pairs and select the top-512.

Used for sources whose reference images are already curated upstream
(**Pick-a-Pic v2**, **Civitai-top**) — they don't need the colorfulness
filter that HPDv3 does. Two curation steps:

  1. discard negative — keep pairs where PickScore(reference) -
                        PickScore(fake) > 0.02.
  2. top-K selection  — sort by PickScore(reference) descending, take the
                        top ``top_k`` (default 512).

To run for a different source, change the path constants below
(`prompt_csv`, `pickscore_csv`, `real_image_dir`, `fake_image_dir`,
`output_csv`) — the filter logic is source-agnostic.

Inputs (top-level constants):

  * ``prompt_csv``    — uid+prompt CSV (one row per reference image)
  * ``pickscore_csv`` — score CSV from ``data_curation/score/pickscore.py``
  * ``real_image_dir`` / ``fake_image_dir`` — to record absolute paths

Output schema matches ``filter/hpdv3.py`` minus the colorfulness columns:

    uid, prompt, real_image_path, fake_image_path,
    pickscore_real, pickscore_fake
"""

import os

import pandas as pd


# ---- swap these for Pick-a-Pic v2 / Civitai-top respectively ----
prompt_csv     = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v2/uid_prompt.csv"
pickscore_csv  = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/pick_a_pic_v2/pickscore/pickscore_score.csv"
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/pick_a_pic_v2/real"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/pick_a_pic_v2/fake"
output_csv     = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/pick_a_pic_v2/filtered_top512.csv"

pickscore_gap_threshold = 0.02
top_k                   = 512


def main():
    prompts = pd.read_csv(prompt_csv, dtype=str)[["uid", "prompt"]]
    pscore = pd.read_csv(pickscore_csv).rename(columns={
        "real_image_score": "pickscore_real", "fake_image_score": "pickscore_fake",
    })

    df = prompts.merge(pscore, on="uid")
    print(f"joined: {len(df)} pairs (across prompt / pickscore CSVs)")

    # 1. discard negative (pickscore gap)
    df = df[df["pickscore_real"] - df["pickscore_fake"] > pickscore_gap_threshold]
    print(f"after discard negative (pickscore gap > {pickscore_gap_threshold}): {len(df)}")

    # 2. top-K by reference (real) PickScore
    df = df.sort_values("pickscore_real", ascending=False).head(top_k).reset_index(drop=True)
    print(f"after top-{top_k} selection (by pickscore_real desc): {len(df)}")

    df["real_image_path"] = df["uid"].map(lambda u: os.path.join(real_image_dir, f"{u}.png"))
    df["fake_image_path"] = df["uid"].map(lambda u: os.path.join(fake_image_dir, f"{u}.png"))

    out = df[[
        "uid", "prompt", "real_image_path", "fake_image_path",
        "pickscore_real", "pickscore_fake",
    ]]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"saved {len(out)} rows to {output_csv}")


if __name__ == "__main__":
    main()
