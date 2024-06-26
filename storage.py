'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/storage.py
'''

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)


    def to(self, device):
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)


    def insert(self, step, reward, mask):
        self.masks[step + 1].copy_(mask)
        self.rewards[step + 1].copy_(reward)


    def after_update(self):
        self.masks[0].copy_(self.masks[-1])
        self.rewards[0].copy_(self.rewards[-1])


    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0) - 1)):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step + 1]