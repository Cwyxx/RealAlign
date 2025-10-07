code_block_type = "single_image_correlation"
target_prompt = "photo of Ford Focus RS, night time, city, city roads, Miami streets"
reward_model_list = ["pickscore", "imagereward", "clipscore", "clip_iqa", "deqa", "aesthetic", "aesthetic_v2_5", "vila_score", "code" ]

if code_block_type == "show_image":
    ### show image ###
    import os
    import pandas as pd
    from scipy.stats import spearmanr
    import seaborn as sns
    from PIL import Image
    import matplotlib.pyplot as plt

    base_image_dir = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_1.0/pickscore-analysis/SD3.5M-DiffusionNFT-MultiReward/ckpt-0/images"
    base_reward_score_dir = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_1.0/pickscore-analysis/SD3.5M-DiffusionNFT-MultiReward/ckpt-0/reward_score"
    # reward_model_list = [ "pickscore", "hpsv2", "imagereward", "code" ]

    reward_model_df_dict = {}
    first_image_names = None
    all_data = []
    font_size = 8
    label_font_size = 16
    for reward_model in reward_model_list:
        reward_model_path = os.path.join(base_reward_score_dir, f"{reward_model}.csv")
        reward_model_df = pd.read_csv(reward_model_path)
        reward_model_df = reward_model_df.sort_values(by="image_name", ascending=True, ignore_index=True)
        all_data.append(reward_model_df[['prompt', 'image_name', reward_model]])

    total_df = all_data[0]
    for reward_model_idx, reward_model in enumerate(reward_model_list):
        if reward_model_idx == 0: continue
        total_df = pd.merge(total_df, all_data[reward_model_idx][['prompt', 'image_name', reward_model]], on=['prompt', 'image_name'], how='outer')

    grouped_by_prompt = total_df.groupby('prompt')
    for idx, (prompt, group) in enumerate(grouped_by_prompt):
        if prompt != target_prompt: continue
        
        num_top_bottom, num_reward_models = 8, len(reward_model_list)
        num_cols, num_rows = num_top_bottom * 2, len(reward_model_list)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3.1), dpi=300)
        
        for col_idx in range(num_cols):
            for row_idx, reward_model in enumerate(reward_model_list):
                ax = axes[row_idx][col_idx]
                sorted_df = group.sort_values(by=reward_model, ascending=False, ignore_index=True)
                
                if col_idx < num_top_bottom:
                    row_data = sorted_df.loc[col_idx]
                    if row_idx == 0:
                        ax.set_title(f"Rank #{col_idx + 1}", fontsize=label_font_size)
                else:
                    bottom_idx = col_idx - num_top_bottom
                    row_data = sorted_df.tail(num_top_bottom).iloc[bottom_idx]
                    if row_idx == 0:
                        total_images = len(group)
                        rank = total_images - (num_top_bottom - 1) + bottom_idx
                        ax.set_title(f"Rank #{rank}", fontsize=label_font_size, color='red')
                
                image_name = row_data['image_name']
                score = row_data[reward_model]
                image_path = os.path.join(base_image_dir, f"{image_name}.png")
                ax.imshow(Image.open(image_path))
                ax.set_xlabel(f"image_name: {image_name}, score: {score:.2f}", fontsize=font_size)

                if col_idx == 0: ax.set_ylabel(reward_model, fontsize=label_font_size)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # fig.suptitle(f'Prompt: {prompt}', fontsize=12, y=1.002)
        plt.tight_layout()
        plt.savefig(f"{code_block_type}.png")
        plt.show()
        
elif code_block_type == "single_image_correlation":
    ### Grouped correlation analysis ###
    import os
    import pandas as pd
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import seaborn as sns

    base_reward_score_dir = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_1.0/pickscore-analysis/SD3.5M-DiffusionNFT-MultiReward/ckpt-0/reward_score"
    # reward_model_list = ["clip_iqa", "aesthetic_v2_5", "code"]

    reward_model_df_dict = {}
    first_image_names = None
    all_data = []

    for reward_model in reward_model_list:
        reward_model_path = os.path.join(base_reward_score_dir, f"{reward_model}.csv")
        reward_model_df = pd.read_csv(reward_model_path)
        reward_model_df = reward_model_df.sort_values(by="image_name", ascending=True, ignore_index=True)
        all_data.append(reward_model_df[['prompt', 'image_name', reward_model]])
        # print(f"reward_model: {reward_model}")
        # print(f"Columns for {reward_model_df[reward_model].iloc[0]}:", reward_model_df.columns)
        
    total_df = all_data[0]
    for reward_model_idx, reward_model in enumerate(reward_model_list):
        if reward_model_idx == 0: continue
        total_df = pd.merge(total_df, all_data[reward_model_idx][['prompt', 'image_name', reward_model]], on=['prompt', 'image_name'], how='outer')

    grouped_by_prompt = total_df.groupby('prompt')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for idx, (prompt, group) in enumerate(grouped_by_prompt):
        if prompt == target_prompt: 
            scores = group[reward_model_list].values 
            corr_matrix, _ = spearmanr(scores, axis=0)
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=reward_model_list, yticklabels=reward_model_list, vmin=-1, vmax=1, ax=ax, fmt=".2f")
            ax.set_title(f'Spearman Rank Correlation for Prompt:\n{prompt}')
            break
        
    plt.tight_layout()
    plt.savefig(f"{code_block_type}.png")
    plt.show()
