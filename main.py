'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/policy.py
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

from armin_pytorch.device import detect_device
from armin_utils.os import current_file_dir

from DVRL.DVRL_policy import DVRL_policy
from DVRL.storage import RolloutStorage

device = detect_device()

num_processes = 16
num_steps = 5

action_space = envs.action_space
nr_inputs = envs.observation_space.shape[0]

actor_critic = DVRLPolicy(action_space, nr_inputs, **model_params)

obs_shape = None

rollouts = RolloutStorage(num_steps=num_steps5, 
                          num_processes=num_processes, 
                          obs_shape=obs_shape, 
                          action_space=envs.action_space)
    
current_obs = torch.zeros(num_processes, *obs_shape)
    
obs = env.reset()
    
if not actor_critic.observation_type == 'fc':
   obs = obs / 255.
    
current_obs = torch.from_numpy(obs).float()
init_states = actor_critic.new_latent_state()
init_rewards = torch.zeros([rl_setting['num_processes'], 1])

if envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = envs.action_space.shape[0]
    init_actions = torch.zeros(rl_setting['num_processes'], action_shape)

    init_states = init_states.to(device)
    init_actions = init_actions.to(device)
    current_obs = current_obs.to(device)
    init_rewards = init_rewards.to(device)
    actor_critic.to(device)
    rollouts.to(device)

    curr_memory = {
        'current_obs': current_obs,
        'states': init_states,
        'oneHotActions': utils.toOneHot(envs.action_space, init_actions),
        'rewards': init_rewards}
    
optimizer = optim.Adam(actor_critic.parameters(), opt['lr'], eps=opt['eps'], betas=opt['betas'])
        

for j in range(1000):
    for step in range(5):
        old_obs = curr_memory['curr_obs']

        policy_return = actor_critic(curr_memory=curr_memory, pred_times=pred_times)
        cpu_actions = policy_return.action.detach().squeeze(1).cpu().numpy()
        
        obs, reward, done, info = envs.step(cpu_actions)
        
        if not actor_critic.observation_type == 'fc':
            obs = obs / 255.
        
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        masks = masks.to(device)
        masks = masks.type(policy_return.action.type())
        
        curr_memory['curr_obs'] = torch.from_numpy(obs).float()
        curr_memory['states'] = actor_critic.vec_conditional_new_latent_state(policy_return.latent_state, masks)
        
        curr_memory['oneHotActions'] = utils.toOneHot(envs.action_space, policy_return.action*masks)
        
        curr_memory['rewards'][:] = reward
        
            
        rollouts.insert(step, reward, masks)
    
        tracked_values = track_values(tracked_values, policy_return)

        if policy_return.predicted_obs_img is not None:
            save_images(policy_return, old_observation, id_tmp_dir, j, step)

        final_rewards, avg_nr_observed, num_ended_episodes = track_rewards(
            tracked_rewards, reward, masks, blank_mask)

        with torch.no_grad():
            policy_return = actor_critic(curr_memory=curr_memory, pred_times=pred_times)
        
        next_value = policy_return.value_estimate

        rollouts.compute_returns(next_value, rl_setting['gamma'])

        values = torch.stack(tuple(tracked_values['values']), dim=0)
        action_log_probs = torch.stack(tuple(tracked_values['action_log_probs']), dim=0)
        
        
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        avg_encoding_loss = torch.stack(tuple(tracked_values['encoding_loss'])).mean()
        dist_entropy = torch.stack(tuple(tracked_values['dist_entropy'])).mean()

        total_loss = (value_loss * loss_function['value_loss_coef']
                      + action_loss * loss_function['action_loss_coef']
                      - dist_entropy * loss_function['entropy_coef']
                      + avg_encoding_loss * loss_function['encoding_loss_coef'])

        optimizer.zero_grad()

        retain_graph = j % algorithm['multiplier_backprop_length'] != 0
        total_loss.backward(retain_graph=retain_graph)

        if opt['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(actor_critic.parameters(), opt['max_grad_norm'])

        optimizer.step()