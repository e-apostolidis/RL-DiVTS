# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def aesthetics_reward(aesthetic_scores, selections, num_of_picks):
    """ Computes the average aesthetic score for the collection of selected frames.

    :param torch.Tensor aesthetic_scores: A tensor of aesthetic scores with shape [1, T]
    :param torch.Tensor selections: Binary valued tensor, contains the selected (1) and non-selected frames e.g. [1, 0, 0, 1]
    :param int num_of_picks: The number of selected thumbnails
    :return: A (float) scalar that represents the aesthetics reward
    """
    aesthetic_scores = aesthetic_scores.squeeze(0)
    masked_aesthetic_scores = aesthetic_scores * selections
    total_aes_reward = torch.sum(masked_aesthetic_scores)
    aes_reward = total_aes_reward / num_of_picks

    return aes_reward


def diversity_reward(image_features, selections):
    """ Computes the average diversity score for the collection of selected frames.

    :param torch.Tensor image_features: Tensor of shape [T, input_size] containing the frame features produced by
           using the pool5 layer of GoogleNet.
    :param torch.Tensor selections: Binary valued tensor, contains the selected (1) and non-selected frames e.g. [1, 0, 0, 1]
    :return: A (float) scalar that represents the diversity reward
    """
    actual_picks = torch.count_nonzero(selections)
    sel = selections.view(-1, 1)
    masked_image_features = torch.mul(image_features, sel)

    # Compute the intra-group similarity
    x_unit = F.normalize(masked_image_features, p=2, dim=1)
    similarity = x_unit @ x_unit.t()
    similarity.fill_diagonal_(0.)
    mean_similarity = similarity.sum() / (actual_picks * (actual_picks - 1))

    div_reward = 1 - mean_similarity

    return div_reward


def representativeness_reward(image_features, selections):
    """ Computes the representativeness score for the collection of selected frames.

    :param torch.Tensor image_features: Tensor of shape [T, input_size] containing the frame features produced by
           using the pool5 layer of GoogleNet.
    :param list[torch.Tensor] selections: Contains the indices of the selected frames
    :return: A (float) scalar that represents the representativeness reward
    """

    selected_image_features = image_features[torch.stack(selections)]
    distance_matrix = torch.cdist(image_features, selected_image_features, 2, 'donot_use_mm_for_euclid_dist')

    min_distance = torch.min(distance_matrix, 1)
    mean_distance = torch.mean(min_distance[0])

    rep_reward = torch.exp(-mean_distance)

    return rep_reward
