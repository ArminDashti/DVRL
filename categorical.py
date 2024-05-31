'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/policy.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))


    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)


    def forward(self, x):
        x = self.linear(x)
        return x


    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x, dim=-1)
        if deterministic is False:
            action = probs.multinomial(1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action


    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy