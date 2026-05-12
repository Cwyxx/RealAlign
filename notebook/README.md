# Notebook Artifacts

This directory contains only the analysis notebooks and exported figure artifacts used for the RealAlign ICML 2026 paper.

## Layout

| Directory | Purpose |
|---|---|
| [`diversity/`](diversity/) | Qwen3.5-27B semantic category analysis for top-512 PickScore real images. |
| [`figures/complementarity/`](figures/complementarity/) | Bar charts for "Complementarity with Existing Preference Alignment Methods". |
| [`figures/human_preference_vs_stylization/`](figures/human_preference_vs_stylization/) | Human-preference vs. stylization comparison on SD-3.5-M. |
| [`figures/detail_vs_realism/`](figures/detail_vs_realism/) | Realism vs. texture-detail comparison on SD-1.5. |
| [`figures/user_study/`](figures/user_study/) | User-study result figure and supporting notebook. |
| [`figures/training_pairs_ablation/`](figures/training_pairs_ablation/) | Training-pair-count ablation figure. |
| [`figures/qualitative/`](figures/qualitative/) | Main-paper and appendix qualitative visualization figures. |

## Diversity Analysis

| Path | Purpose |
|---|---|
| [`diversity/qwen_semantic_diversity_evaluation.py`](diversity/qwen_semantic_diversity_evaluation.py) | Classifies each real image into one semantic category with Qwen3.5-27B. |
| [`diversity/summarize_diversity_results.py`](diversity/summarize_diversity_results.py) | Summarizes the category distribution. |
| [`diversity/run_qwen_semantic_diversity.sh`](diversity/run_qwen_semantic_diversity.sh) | Launcher for the diversity evaluation. |
| [`diversity/results/top_512_pickscore_real_image_categories.csv`](diversity/results/top_512_pickscore_real_image_categories.csv) | Per-image semantic category results. |
| [`diversity/results/diversity_summary.csv`](diversity/results/diversity_summary.csv) | Aggregated diversity summary. |

## Figure Sources

| Path | Output |
|---|---|
| [`figures/complementarity/complementarity_bar_charts.ipynb`](figures/complementarity/complementarity_bar_charts.ipynb) | `sd15_*_bar_chart.*`, `sd35m_*_bar_chart.*`; also saves the user-study plot into `../user_study/`. |
| [`figures/human_preference_vs_stylization/human_preference_vs_stylization.ipynb`](figures/human_preference_vs_stylization/human_preference_vs_stylization.ipynb) | `human_preference_vs_stylization.{pdf,svg}`. |
| [`figures/detail_vs_realism/sd15_detail_vs_realism.ipynb`](figures/detail_vs_realism/sd15_detail_vs_realism.ipynb) | `sd15_detail_vs_realism.{pdf,svg}`. |
| [`figures/user_study/user_study.ipynb`](figures/user_study/user_study.ipynb) | Supports `user_study.{pdf,svg}`. |
| [`figures/training_pairs_ablation/training_pairs_ablation.ipynb`](figures/training_pairs_ablation/training_pairs_ablation.ipynb) | `training_pairs_ablation.pdf`. |
| [`figures/qualitative/sd15_main_qualitative.ipynb`](figures/qualitative/sd15_main_qualitative.ipynb) | `sd15_main_qualitative.pdf`. |
| [`figures/qualitative/sd35m_main_qualitative.ipynb`](figures/qualitative/sd35m_main_qualitative.ipynb) | `sd35m_main_qualitative.pdf`. |
| [`figures/qualitative/appendix_qualitative.ipynb`](figures/qualitative/appendix_qualitative.ipynb) | `sd15_appendix_qualitative.pdf`, `sd35m_appendix_qualitative.pdf`, and related appendix exports. |

