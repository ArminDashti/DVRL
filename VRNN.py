import torch
import torch.nn as nn
import torch.nn.functional as F
from DVRL import encoder_decoder


class VRNN_encoding(nn.Module):
    def __init__(self, phi_x_dim, nr_actions, action_encoding, observation_type, nr_inputs, cnn_channels, encoder_batch_norm):
        super().__init__()
        self.action_encoding = action_encoding
        self.phi_x_dim = phi_x_dim
        assert(action_encoding > 0)

        self.phi_x = encoder_decoder.get_encoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm)

        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type,
            cnn_channels)
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        if encoder_batch_norm:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding),
                nn.BatchNorm1d(action_encoding),
                nn.ReLU())
        else:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding),
                nn.ReLU())
        self.nr_actions = nr_actions

    def forward(self, observation_states):
        seq_len, batch_size, *obs_dim = observation_states.all_x.size()

        all_phi_x = self.phi_x(observation_states.all_x.view(-1, *obs_dim)).view(-1, self.cnn_output_number)

        all_phi_x = all_phi_x.view(seq_len, batch_size, -1)
        observation_states.all_phi_x = all_phi_x

        if self.action_encoding > 0:
            encoded_action = self.action_encoder(
                observation_states.all_a.view(-1, self.nr_actions)
                ).view(seq_len, batch_size, -1)
            observation_states.encoded_action = encoded_action

        return observation_states
    
    
class VRNN_transition(nn.Module):
    def __init__(self, h_dim, z_dim, action_encoding):
        super().__init__()
        self.prior = nn.Sequential(
            nn.Linear(h_dim + action_encoding, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.action_encoding = action_encoding


    def forward(self, previous_latent_state, observation_states):
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        if self.action_encoding > 0:
            input = torch.cat([
                previous_latent_state.h,
                observation_states.encoded_action], 2).view(-1, h_dim + self.action_encoding)
        else:
            input = previous_latent_state.h.view(-1, h_dim)

        prior_t = self.prior(input)

        prior_mean_t = self.prior_mean(prior_t).view(batch_size, num_particles, -1)
        prior_std_t = self.prior_std(prior_t).view(batch_size, num_particles, -1)

        prior_dist = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(
                mean=prior_mean_t,
                variance=prior_std_t))

        return prior_dist


class VRNN_deterministic_transition(nn.Module):
    def __init__(self, z_dim, phi_x_dim, h_dim, action_encoding):
        super().__init__()
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())
        self.rnn = nn.GRUCell(h_dim + phi_x_dim + action_encoding, h_dim)
        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, latent_state, observation_states, time):
        batch_size, num_particles, z_dim = latent_state.z.size()
        batch_size, num_particles, phi_x_dim = observation_states.all_phi_x.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        phi_x = observation_states.all_phi_x

        phi_z_t = self.phi_z(latent_state.z.view(-1, z_dim)).view(batch_size, num_particles, h_dim)

        if self.action_encoding > 0:
            input = torch.cat([
                phi_x,
                phi_z_t,
                observation_states.encoded_action], 2).view(-1, phi_x_dim + h_dim + self.action_encoding)
        else:
            input = torch.cat([
                phi_x,
                phi_z_t], 1).view(-1, phi_x_dim + h_dim)

        h = self.rnn(input, previous_latent_state.h.view(-1, h_dim))

        latent_state.phi_z = phi_z_t.view(batch_size, num_particles, -1)
        latent_state.h = h.view(batch_size, num_particles, h_dim)
        return latent_state
    
    
class VRNN_emission(nn.Module):
    def __init__(self, h_dim, action_encoding, observation_type, nr_inputs,
                 cnn_channels, encoder_batch_norm):
        super().__init__()
        self.observation_type = observation_type
        self.action_encoding = action_encoding

        encoding_dimension = h_dim + h_dim + action_encoding

        self.dec, self.dec_mean, self.dec_std = encoder_decoder.get_decoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm)

        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type,
            cnn_channels)
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        if encoder_batch_norm:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number),
                nn.BatchNorm1d(self.cnn_output_number),
                nn.ReLU())
        else:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number),
                nn.ReLU())


    def forward(self, previous_latent_state, latent_state, observation_states):
        batch_size, num_particles, phi_z_dim = latent_state.phi_z.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        dec_t = self.linear_obs_decoder(torch.cat([
            latent_state.phi_z,
            previous_latent_state.h,
            observation_states.encoded_action
        ], 2).view(-1, phi_z_dim + h_dim + self.action_encoding))

        dec_t = self.dec(dec_t.view(-1, *self.cnn_output_dimension))

        dec_mean_t = self.dec_mean(dec_t)
        _, *obs_dim = dec_mean_t.size()
        dec_mean_t = dec_mean_t.view(batch_size, num_particles, *obs_dim)

        if self.observation_type == 'fc':
            dec_std_t = self.dec_std(dec_t).view(batch_size, num_particles, *obs_dim)
            emission_dist = rv.StateRandomVariable(
                all_x=rv.MultivariateIndependentNormal(
                    mean=dec_mean_t,
                    variance=dec_std_t
                )
            )
        else:
            emission_dist = rv.StateRandomVariable(
                all_x=rv.MultivariateIndependentPseudobernoulli(
                    probability=dec_mean_t
                )
            )

        return emission_dist
    
    
class VRNN_proposal(nn.Module):
    def __init__(self, z_dim, h_dim, phi_x_dim, action_encoding, encoder_batch_norm):
        super().__init__()

        if encoder_batch_norm:
            self.enc = nn.Sequential(
                nn.Linear(h_dim + phi_x_dim + action_encoding, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU())
        else:
            self.enc = nn.Sequential(
                nn.Linear(h_dim + phi_x_dim + action_encoding, h_dim),
                nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, observation_states, time):
        batch_size, num_particles, phi_x_dim = observation_states.all_phi_x.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        if self.action_encoding > 0:
            input = torch.cat([
                observation_states.all_phi_x,
                previous_latent_state.h,
                observation_states.encoded_action
            ], 2).view(-1, phi_x_dim + h_dim + self.action_encoding)
        else:
            input = torch.cat([
                observation_states.all_phi_x,
                previous_latent_state.h
            ], 2).view(-1, phi_x_dim + h_dim)

        enc_t = self.enc(input)

        enc_mean_t = self.enc_mean(enc_t).view(batch_size, num_particles, -1)
        enc_std_t = self.enc_std(enc_t).view(batch_size, num_particles, -1)

        proposed_state = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(
                mean=enc_mean_t,
                variance=enc_std_t
                ))
        return proposed_state