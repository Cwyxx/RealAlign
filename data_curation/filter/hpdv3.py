"""Filter HPDv3 (reference, fake) pairs and select the top-512 for training.

Four curation steps applied in order:

  1. anime drop       — drop rows whose reference image was classified as
                        artwork / anime by ``score/anime.py`` (HPDv3 only;
                        the ``real_images`` split contains illustrations
                        that aren't photographic).
  2. color filter     — keep pairs where colorfulness(reference) >
                        colorfulness(fake) by ``color_gap_threshold``
                        (default 0).
  3. discard negative — keep pairs where PickScore(reference) -
                        PickScore(fake) > 0.02 (the paper's threshold).
  4. top-K selection  — sort the remaining pairs by PickScore(reference)
                        descending and take the top ``top_k`` (default 512).

Inputs (top-level constants below):

  * ``prompt_csv``       — uid+prompt CSV from ``data_curation/extract/hpdv3.py``
  * ``anime_csv``        — classification CSV from ``data_curation/score/anime.py``
  * ``colorfulness_csv`` — score CSV from ``data_curation/score/colorfulness.py``
  * ``pickscore_csv``    — score CSV from ``data_curation/score/pickscore.py``
  * ``real_image_dir`` / ``fake_image_dir`` — to record absolute paths in the
    output

Output: a CSV with one row per selected pair, ready to be consumed by the
Stage 1 / Stage 2 dataloader. Columns:

    uid, prompt, real_image_path, fake_image_path,
    pickscore_real, pickscore_fake, color_real, color_fake
"""

import os

import pandas as pd


prompt_csv       = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real_images_uid_prompt.csv"
anime_csv        = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/anime/anime_classification.csv"
colorfulness_csv = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/colorfulness/colorfulness_score.csv"
pickscore_csv    = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/pickscore/pickscore_score.csv"
real_image_dir   = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real"
fake_image_dir   = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/fake"
output_csv       = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/filtered_top512.csv"

color_gap_threshold     = 0.0   # real should be more colorful than fake
pickscore_gap_threshold = 0.02  # paper's threshold
top_k                   = 512


def main():
    prompts = pd.read_csv(prompt_csv, dtype=str)[["uid", "prompt"]]
    color = pd.read_csv(colorfulness_csv).rename(columns={
        "real_image_score": "color_real", "fake_image_score": "color_fake",
    })
    pscore = pd.read_csv(pickscore_csv).rename(columns={
        "real_image_score": "pickscore_real", "fake_image_score": "pickscore_fake",
    })

    df = prompts.merge(color, on="uid").merge(pscore, on="uid")
    print(f"joined: {len(df)} pairs (across prompt / colorfulness / pickscore CSVs)")

    # 1. anime drop — references classified as artwork/anime are not photographic
    anime = pd.read_csv(anime_csv, dtype=str)
    non_anime_uids = set(anime.loc[anime["anime"].str.lower() == "no", "uid"])
    df = df[df["uid"].isin(non_anime_uids)]
    print(f"after anime drop (anime == 'no'): {len(df)}")

    # 2. color filter
    df = df[df["color_real"] - df["color_fake"] > color_gap_threshold]
    print(f"after color filter (gap > {color_gap_threshold}): {len(df)}")

    # 3. discard negative (pickscore gap)
    df = df[df["pickscore_real"] - df["pickscore_fake"] > pickscore_gap_threshold]
    print(f"after discard negative (pickscore gap > {pickscore_gap_threshold}): {len(df)}")

    # 4. top-K by reference PickScore
    df = df.sort_values("pickscore_real", ascending=False).head(top_k).reset_index(drop=True)
    print(f"after top-{top_k} selection (by pickscore_real desc): {len(df)}")

    df["real_image_path"] = df["uid"].map(lambda u: os.path.join(real_image_dir, f"{u}.png"))
    df["fake_image_path"] = df["uid"].map(lambda u: os.path.join(fake_image_dir, f"{u}.png"))

    out = df[[
        "uid", "prompt", "real_image_path", "fake_image_path",
        "pickscore_real", "pickscore_fake", "color_real", "color_fake",
    ]]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"saved {len(out)} rows to {output_csv}")


if __name__ == "__main__":
    main()
