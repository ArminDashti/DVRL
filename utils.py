'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/aesmc/inference.py
'''


import numpy as np
import torch


def sample_ancestral_index(log_weight):
    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = np.exp(math.lognormexp(log_weight.cpu().detach().numpy(), dim=1))

    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    cumulative_weights = cumulative_weights / np.max(cumulative_weights, axis=1, keepdims=True)

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    temp = torch.from_numpy(indices).long().to(device)

    return temp