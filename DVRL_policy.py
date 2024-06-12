'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/policy.py
'''


import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import aesmc.random_variable as rv
import aesmc.state as st
import aesmc.util as ae_util
import aesmc.statistics as stats
import aesmc.math as math
import aesmc.test_utils as tu
from aesmc.inference import sample_ancestral_index
from DVRL import encoder_decoder, policy
import numpy as np
from operator import mul
from functools import reduce


def num_killed_particles(A):
    batch_size, num_particles = A.size()
    output = np.zeros(batch_size)

    for batch in range(batch_size):
        output[batch] = np.count_nonzero(np.bincount(A[batch].numpy(), minlength=num_particles) == 0)

    return output.astype(int)


class DVRL_policy(policy):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 observation_type,
                 action_encoding,
                 cnn_channels,
                 h_dim,
                 init_function,
                 encoder_batch_norm,
                 policy_batch_norm,
                 prior_loss_coef,
                 obs_loss_coef,
                 detach_encoder,
                 batch_size,
                 num_particles,
                 particle_aggregation,
                 z_dim,
                 resample
                 ):
        super().__init__(action_space, encoding_dimension=h_dim)
        self.init_function = init_function
        self.num_particles = num_particles
        self.particle_aggregation = particle_aggregation
        self.batch_size = batch_size
        self.obs_loss_coef = float(obs_loss_coef)
        self.prior_loss_coef = float(prior_loss_coef)
        self.observation_type = observation_type
        self.encoder_batch_norm = encoder_batch_norm
        self.policy_batch_norm = policy_batch_norm
        self.detach_encoder = detach_encoder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.resample = resample


        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type,
            cnn_channels)
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        phi_x_dim = self.cnn_output_number

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
        else:
            action_shape = action_space.shape[0]

        self.encoding_network = VRNN_encoding(
            phi_x_dim=phi_x_dim,
            nr_actions=action_shape,
            action_encoding=action_encoding,
            obs_type=obs_type,
            nr_inputs=nr_inputs,
            cnn_channels=cnn_channels,
            encoder_batch_norm=encoder_batch_norm
            )

        # Computes p(z_t|h_{t-1}, a_{t-1})
        self.transition_network = VRNN_transition(
            h_dim=h_dim,
            z_dim=z_dim,
            action_encoding=action_encoding)

        # Computes h_t=f(h_{t-1}, z_t, a_{t-1}, o_t)
        self.deterministic_transition_network = VRNN_deterministic_transition(
            z_dim=z_dim,
            phi_x_dim=phi_x_dim,
            h_dim=h_dim,
            action_encoding=action_encoding
            )

        # Computes p(o_t|h_t, z_t, a_{t-1})
        self.emission_network = VRNN_emission(
            h_dim=h_dim,
            action_encoding=action_encoding,
            observation_type=observation_type,
            nr_inputs=nr_inputs,
            cnn_channels=cnn_channels,
            encoder_batch_norm=encoder_batch_norm
            )

        # Computes q(z_t|h_{t-1}, a_{t-1}, o_t)
        self.proposal_network = VRNN_proposal(
            z_dim=z_dim,
            h_dim=h_dim,
            phi_x_dim=phi_x_dim,
            action_encoding=action_encoding,
            encoder_batch_norm=encoder_batch_norm
            )

        dim = 2 * h_dim + 1
        if particle_aggregation == 'rnn' and self.num_particles > 1:
            self.particle_gru = nn.GRU(dim, h_dim, batch_first=True)

        elif self.num_particles == 1:
            self.particle_gru = nn.Linear(dim, h_dim)

        self.reset_parameters()


    def new_latent_state(self):
        h = torch.zeros(self.bs, self.num_particles, self.h_dim).to(device)
        init_state = st.State(h=h)

        log_weight = torch.zeros(self.batch_size, self.num_particles).to(device)

        init_state.log_weight = log_weight

        return init_state


    def vec_condition_new_latent(self, latent_state, mask):
        return latent_state.multiply_each(mask, only=['log_weight', 'h', 'z'])


    def reset_parameters(self):
        def weights_init(gain):
            def fn(m):
                classname = m.__class__.__name__
                init_func = getattr(torch.nn.init, self.init_function)
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    init_func(m.weight.data, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                if classname.find('GRUCell') != -1:
                    init_func(m.weight_ih.data)
                    init_func(m.weight_hh.data)
                    m.bias_ih.data.fill_(0)
                    m.bias_hh.data.fill_(0)

            return fn

        relu_gain = nn.init.calculate_gain('relu')
        self.apply(weights_init(relu_gain))
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)


    def sample_from(self, state_rand_var):
        return state_rand_var.sample_reparameterized(self.batch_size, self.num_particles)


    def encode(self, obs, reward, actions, pre_latent, pred_times):
        bs, *rest = obs.size()
        img_obs = obs.unsqueeze(0).contiguous()
        actions = actions.unsqueeze(0).contiguous()
        reward = reward.unsqueeze(0).contiguous()
        
        obs_states = st.State(all_x=img_obs, all_a=actions, r=reward)

        old_log_weight = pre_latent.log_weight

        obs_states = self.encoding_network(obs_states) # Encode observation_states by a CNN network.

        obs_states.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)

        ancestral_indices = sample_ancestral_index(old_log_weight)

        num_killed_particles = list(num_killed_particles(ancestral_indices.data.cpu()))
        
        if self.resample:
            previous_latent_state = prev_latent.resample(ancestral_indices)
        else:
            num_killed_particles = [0] * batch_size

        avg_num_killed_particles = sum(num_killed_particles)/len(num_killed_particles)

        curr_obs = obs_states.index_elements(0)
        
        # Take prev latent and obs and calcualte dist, mean, std with 3 indivisual networks then feed rv.MultivariateIndependentNormal(mean=enc_mean_t, variance=enc_std_t) to rv.StateRandomVariable(z=z)
        proposal_state_rand_var = self.proposal_network(prev_latent=prev_latent, obs_states=curr_obs, time=0) 

        latent = self.sample_from(proposal_state_rand_var)

        latent = self.deterministic_transition_network(prev_latent=prev_latent, latent=latent, obs_states=obs_states, time=0)

        transition_state_rand_var = self.transition_network(prev_latent, curr_obs)

        emission_state_rand_var = self.emission_network(prev_latent, latent, curr_obs)

        emission_logpdf = emission_state_rand_var.logpdf(curr_obs, bs, self.num_particles)

        proposal_logpdf = proposal_state_rand_var.logpdf(latent_state, batch_size, self.num_particles)
        
        transition_logpdf = transition_state_rand_var.logpdf(latent_state, batch_size, self.num_particles)

        new_log_weight = transition_logpdf - proposal_logpdf + emission_logpdf

        latent_state.log_weight = new_log_weight

        encoding_logli = math.logsumexp(new_log_weight, dim=1) - np.log(self.num_particles)

        pred_obs = None
        particle_obs = None
        
        if predicted_times is not None:
            pred_obs, particle_obs = self.pred_obs(
                latent_state=latent_state,
                current_obs=current_obs,
                actions=actions,
                emission_state_random_variable=emission_state_random_variable,
                predicted_times=predicted_times)

        ae_util.init(False)

        return latent_state, \
            - encoding_logli, \
            (- transition_logpdf + proposal_logpdf, - emission_logpdf),\
            avg_num_killed_particles,\
            pred_obs, particle_obs


    def pred_obs(self, latent_state, current_observation, actions, emission_state_random_variable, predicted_times):
        max_distance = max(predicted_times)
        old_log_weight = latent_state.log_weight
        predicted_observations = []
        particle_observations = []

        if 0 in predicted_times:
            x = emission_state_random_variable.all_x._probability
            averaged_obs = stats.empirical_mean(x, old_log_weight)
            predicted_observations.append(averaged_obs)
            particle_observations.append(x)

        bs, num_particles, z_dim = latent.z.size()
        bs, num_particles, h_dim = latent.h.size()
        
        for dt in range(max_distance):
            old_observation = current_observation
            previous_latent_state = latent_state

            transition_state_random_variable = self.transition_network(previous_latent_state,old_observation)
            latent_state = self.sample_from(transition_state_random_variable)

            latent_state.phi_z = self.deterministic_transition_network.phi_z(latent_state.z.view(-1, z_dim)).view(batch_size, num_particles, h_dim)

            emission_state_random_variable = self.emission_network(previous_latent_state, latent_state, old_observation)
            
            x = emission_state_rand_var.all_x._prob
            
            averaged_obs = stats.empirical_mean(x, old_log_weight)

            current_observation = st.State(all_x=averaged_obs.unsqueeze(0), all_a=actions.contiguous())
            
            current_observation = self.encoding_network(current_observation)
            current_observation.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)
            current_observation = current_observation.index_elements(0)

            latent_state = self.deterministic_transition_network(prev_latent=latent, latent=latent, obs_states=current_obs, time=0)

            if dt+1 in predicted_times:
                pred_obs.append(averaged_obs)
                particle_obs.append(x)

        return pred_obs, particle_obs


    def encode_particles(self, latent_state):
        batch_size, num_particles, h_dim = latent_state.h.size()
        state = torch.cat([latent_state.h, latent_state.phi_z], dim=2)

        normalized_log_weights = math.lognormexp(latent_state.log_weight, dim=1)

        particle_state = torch.cat([state, torch.exp(normalized_log_weights).unsqueeze(-1)], dim=2)

        if self.num_particles == 1:
            particle_state = particle_state.squeeze(1)
            encoded_particles = self.particle_gru(particle_state)
            return encoded_particles
        else:
            _ , encoded_particles = self.particle_gru(particle_state)
            return encoded_particles[0]