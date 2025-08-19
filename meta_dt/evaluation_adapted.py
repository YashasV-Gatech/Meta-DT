# Meta-DT/meta_dt/evaluation_adapted.py
import numpy as np
import torch
from collections import OrderedDict

# Renamed from meta_evaluate_episode_rtg
def meta_evaluate_episode_rtg_adapted(
    env, config, model, context_encoder, 
    state_mean=0.0, state_std=1.0, device='cuda', 
    target_return=None, mode='normal', prompt=None, current_step=0
):
    """
    Adapted evaluation function for Meta-DT integration.
    Uses standardized configuration and environment interface.
    """
    
    # Extract parameters from config
    state_dim = config.state_dim
    action_dim = config.act_dim
    max_episode_steps = config.max_episode_steps
    scale = config.return_scale
    horizon = config.context_horizon # Context horizon (h)
    context_dim = config.context_dim
    num_eval_episodes = config.num_eval_episodes
    warm_train_steps = config.warm_train_steps

    model.eval()
    context_encoder.eval()
    model.to(device=device)
    context_encoder.to(device=device)

    # Prepare normalization constants
    state_mean_t = torch.from_numpy(state_mean).to(device=device)
    state_std_t = torch.from_numpy(state_std).to(device=device)

    avg_epi_return = 0.0
    avg_epi_len = 0

    # Store the last trajectory generated (for the self-guided prompt buffer)
    last_trajectory = None

    for _ in range(num_eval_episodes):
        # MODIFICATION: Use standardized reset
        state, _ = env.reset() 
        
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # Initialize buffers for the episode
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        contexts = torch.zeros((1, context_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        
        # Buffers for context calculation (numpy based)
        states_traj = np.zeros((max_episode_steps, state_dim))
        actions_traj = np.zeros((max_episode_steps, action_dim))
        rewards_traj = np.zeros((max_episode_steps, 1))

        target_returns = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        for t in range(max_episode_steps):
            
            # Add placeholder for current action/reward
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            # Determine if prompt should be used (based on current training step)
            use_prompt = prompt if current_step >= warm_train_steps else None

            # Get action from DT
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean_t) / state_std_t, 
                contexts.to(dtype=torch.float32), 
                actions.to(dtype=torch.float32), 
                rewards.to(dtype=torch.float32), 
                target_returns.to(dtype=torch.float32), 
                timesteps.to(dtype=torch.long), 
                prompt=use_prompt, 
                # MODIFICATION: Pass config instead of args, and current_step instead of epoch
                config=config, 
                current_step=current_step
            )
            
            actions[-1] = action
            action = action.detach().cpu().numpy()

            # MODIFICATION: Use standardized step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update numpy trajectory buffers
            # Note: states_traj[t] should store the state BEFORE the action was taken
            states_traj[t] = np.copy(states[-1].detach().cpu().numpy().reshape(-1))
            actions_traj[t] = np.copy(action)
            rewards_traj[t] = np.copy(reward)

            # Update tensor buffers
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            
            # --- Context Calculation (GRU) ---
            # Prepare segments for the GRU (h steps leading up to the current state)
            # (Logic adapted from original evaluation.py)
            
            # Start index for the segment
            start_idx = max(0, t + 1 - horizon)
            
            state_seg = states_traj[start_idx:t + 1]
            action_seg = actions_traj[start_idx:t + 1]
            reward_seg = rewards_traj[start_idx:t + 1]
            
            # Padding
            length_gap = horizon - state_seg.shape[0]
            state_seg = np.pad(state_seg, ((length_gap, 0), (0, 0)))
            action_seg = np.pad(action_seg, ((length_gap, 0), (0, 0)))
            reward_seg = np.pad(reward_seg, ((length_gap, 0), (0, 0)))
            
            # Convert to tensors and reshape for GRU (Time, Batch, Feature)
            state_seg_t = torch.FloatTensor(state_seg).to(device).unsqueeze(1)
            action_seg_t = torch.FloatTensor(action_seg).to(device).unsqueeze(1)
            reward_seg_t = torch.FloatTensor(reward_seg).to(device).unsqueeze(1)

            # Calculate context
            cur_context = context_encoder(state_seg_t, action_seg_t, reward_seg_t).detach().reshape(1, -1)
            contexts = torch.cat([contexts, cur_context], dim=0)

            # Update rewards and RTG
            rewards[-1] = reward
            pred_return = target_returns[0, -1] - (reward / scale)
            target_returns = torch.cat([target_returns, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            avg_epi_return += reward
            avg_epi_len += 1

            if done:
                break
        
        # Store the trajectory generated in this episode
        # Note: states includes the initial state and the final state (len T+1)
        # actions and rewards have length T
        last_trajectory = OrderedDict([
            ('observations', states[:-1].cpu().detach().numpy()), 
            ('actions', actions.cpu().detach().numpy()), 
            ('rewards', rewards.cpu().detach().numpy()), 
            ('next_observations', states[1:].cpu().detach().numpy())
        ])

    return (avg_epi_return / num_eval_episodes, avg_epi_len / num_eval_episodes, last_trajectory)