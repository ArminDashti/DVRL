import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from policy import Categorical, DiagGaussian
from torch.nn.init import xavier_normal_, orthogonal_
import encoder_decoder
import namedlist
from operator import mul
from functools import reduce


class Policy(nn.Module):
    def __init__(self, action_space, encoding_dimension):
        super().__init__()

        self.critic_linear = nn.Linear(encoding_dimension, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(encoding_dimension, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(encoding_dimension, num_outputs)
        else:
            raise NotImplementedError

        self.encoding_bn = nn.BatchNorm1d(encoding_dimension)


    def forward(self, current_memory, deterministic=False, predicted_times=None):
        policy_return = PolicyReturn()

        device = next(self.parameters()).device

        latent_state, total_encoding_loss, encoding_losses, n_killed_p,\
            img, p_img = self.encode(
                observation=current_memory['current_obs'].to(device),
                reward=current_memory['rewards'].to(device),
                actions=current_memory['oneHotActions'].to(device).detach(),
                previous_latent_state=current_memory['states'].to(device),
                predicted_times=predicted_times,
            )

        latent_state_for_encoding = latent_state.detach() if self.detach_encoder else latent_state

        encoded_state = (self.encode_particles(latent_state_for_encoding)
                         if type(self).__name__ == 'DVRLPolicy'
                         else latent_state_for_encoding)

        if self.policy_batch_norm:
            encoded_state = self.encoding_bn(encoded_state)

        policy_return.latent_state = latent_state
        policy_return.total_encoding_loss = total_encoding_loss
        policy_return.encoding_losses = encoding_losses
        policy_return.num_killed_particles = n_killed_p
        policy_return.predicted_obs_img = img
        policy_return.particle_obs_img = p_img

        policy_return.value_estimate = self.critic_linear(encoded_state)
        action = self.dist.sample(encoded_state, deterministic=deterministic)
        policy_return.action = action

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(encoded_state, action.detach())
        policy_return.action_log_probs = action_log_probs
        policy_return.dist_entropy = dist_entropy

        return policy_return