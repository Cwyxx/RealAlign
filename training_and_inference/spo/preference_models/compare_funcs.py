import torch
from .builder import COMPARE_FUNCS

@COMPARE_FUNCS.register_module()
def preference_score_compare(scores, threshold):
    # scores: num_sample_per_step, b
    scores, indices = torch.sort(scores, dim=0, descending=True)
    # 2, b
    indices = indices[[0, -1], :]
    scores = scores[[0, -1], :]
    scores = scores.softmax(dim=0)
    # b
    valid_samples = scores[0] - scores[1] > threshold
    return indices, valid_samples

@COMPARE_FUNCS.register_module()
def aggregate_rewards_by_rank_compare(scores, threshold, aigi_detector_weight):
    reward_model_scores, aigi_detector_scores = scores
    
    def get_ranks(tensor_scores):
        indices_desc = torch.argsort(tensor_scores, dim=0)
        rank = torch.argsort(indices_desc, dim=0)
        return rank
    
    reward_model_scores_rank = get_ranks(reward_model_scores)
    aigi_detector_scores_rank = get_ranks(aigi_detector_scores)
    weight_rank = (1 - aigi_detector_weight) * reward_model_scores_rank + aigi_detector_weight * aigi_detector_scores_rank
    
    return preference_score_compare(weight_rank, threshold)
    
@COMPARE_FUNCS.register_module()
def spo_reward_aigi_detector_compare(scores, threshold, aigi_detector_weight):
    # def normalize_scores(scores, eps=1e-6):
    #     """
    #     Z-score normalize
    #     """
    #     mean = torch.mean(scores, dim=0, keepdim=True)
    #     std = torch.std(scores, dim=0, keepdim=True)
    #     return (scores - mean) / (std + eps)
    
    def min_max_normalize(tensor: torch.Tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    # reward_model_scores / aigi_detector_scores: num_sample_per_step, b
    reward_model_scores, aigi_detector_scores = scores
    norm_reward = min_max_normalize(reward_model_scores)
    norm_aigi = min_max_normalize(aigi_detector_scores)
    
    # norm_reward = normalize_scores(reward_model_scores)
    # norm_aigi = normalize_scores(aigi_detector_scores)
    
    # weight_scores: num_sample_per_step, b
    weight_scores = (1 - aigi_detector_weight) * norm_reward + aigi_detector_weight * norm_aigi
    
    
    return preference_score_compare(weight_scores, threshold)