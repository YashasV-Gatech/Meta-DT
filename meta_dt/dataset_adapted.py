# experiments/baselines/meta_dt_integration/Meta-DT/meta_dt/dataset_adapted.py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import sys
from pathlib import Path

# Import optimized discount_cumsum from adapter
try:
    from scipy.signal import lfilter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def discount_cumsum(x, gamma=1.0):
    """Optimized discounted cumulative sums."""
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    # Ensure input/output is float32
    x = x.astype(np.float32)

    if SCIPY_AVAILABLE:
        # lfilter computes y[n] = x[n] + gamma*y[n-1].
        # We reverse input/output to calculate cumulative sum starting from the end.
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        # Fallback Numpy implementation
        discount_cumsum_result = np.zeros_like(x, dtype=np.float32)
        discount_cumsum_result[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum_result[t] = x[t] + gamma * discount_cumsum_result[t + 1]
        return discount_cumsum_result

# (Keep append_error_to_trajectory unchanged, ensure it uses the 'device' parameter correctly)
def append_error_to_trajectory(world_model, device, context_horizon, traj, config, mean, std):
    """
    Calculates the World Model prediction error for a given trajectory.
    """
    (context_encoder, dynamics_decoder) = world_model
    # Ensure models are in evaluation mode and on the correct device
    context_encoder.eval()
    dynamics_decoder.eval()
    context_encoder.to(device)
    dynamics_decoder.to(device)

    # Ensure inputs are float32 for processing
    states = traj['observations'].astype(np.float32)
    actions = traj['actions'].astype(np.float32)
    rewards = traj['rewards'].reshape(-1, 1).astype(np.float32)
    next_states = traj['next_observations'].astype(np.float32)

    # ... (Segmentation Logic - CPU) ...
    states_segment, actions_segment, rewards_segment = [], [], []
    horizon = context_horizon
    
    # (Loop for segmentation and padding remains the same)
    for idx in range(states.shape[0]):
        start_idx = max(0, idx - horizon)
        end_idx = idx # History up to (but not including) the current index

        if end_idx == 0:
            state_seg = np.zeros((0, states.shape[1]), dtype=np.float32)
            action_seg = np.zeros((0, actions.shape[1]), dtype=np.float32)
            reward_seg = np.zeros((0, rewards.shape[1]), dtype=np.float32)
        else:
            state_seg = states[start_idx : end_idx]; action_seg = actions[start_idx : end_idx]; reward_seg = rewards[start_idx : end_idx]
            
        # Pre-padding
        length_gap = horizon - state_seg.shape[0]
        states_segment.append(np.pad(state_seg, ((length_gap, 0), (0, 0)), mode='constant'))
        actions_segment.append(np.pad(action_seg, ((length_gap, 0), (0, 0)), mode='constant'))
        rewards_segment.append(np.pad(reward_seg, ((length_gap, 0), (0, 0)), mode='constant'))
    
    # Convert to tensors (Move to target device for WM calculation)
    states_segment_t = torch.from_numpy(np.stack(states_segment, axis=0)).to(device)
    actions_segment_t = torch.from_numpy(np.stack(actions_segment, axis=0)).to(device)
    rewards_segment_t = torch.from_numpy(np.stack(rewards_segment, axis=0)).to(device)
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    
    # 2. Calculate WM predictions and errors (GPU)
    with torch.no_grad():
        is_dynamics_env = "_param" in config.benchmark
        # Encoder expects [seq_len (H), batch_size, dim]
        contexts = context_encoder(
            states_segment_t.transpose(0, 1),
            actions_segment_t.transpose(0, 1),
            rewards_segment_t.transpose(0, 1)
        )
        
        # >>> CRITICAL FIX: Save the calculated contexts back to the trajectory <<<
        traj['contexts'] = contexts.detach().cpu().numpy().astype(np.float32)
        if is_dynamics_env:
            # State prediction error
            # Decoder signature (StateDecoder): forward(self, state, action, reward, next_state, context)
            states_predict = dynamics_decoder(states_t, actions_t, rewards_t, next_states_t, contexts).detach().cpu().numpy()
            # Calculate L1 error (summed across dimensions)
            # We compare against the raw next_states, as the WM predicts raw states.
            traj['errors'] = np.abs(states_predict - next_states).sum(axis=1)
        else:
            # Reward prediction error
            # Decoder signature (RewardDecoder): forward(self, state, action, next_state, context)
            reward_predict = dynamics_decoder(states_t, actions_t, next_states_t, contexts).detach().cpu().numpy()
            traj['errors'] = np.abs(reward_predict - rewards).flatten()
            
    return traj

class MetaDT_Dataset_Adapted(Dataset):
    """
    REFACTORED: PyTorch Dataset for training Meta-DT (Phase 2) using efficient Lazy Loading.
    """
    def __init__(self, trajectories, horizon, max_episode_steps, return_scale, device, prompt_trajectories_list, config, world_model):
        self.trajectories = trajectories # Stored on CPU (Numpy)
        self.horizon = horizon # K
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        # We only need the device reference for WM preprocessing.
        self.target_device = device 
        self.config = config
        self.context_horizon = config.context_horizon # h
        self.prompt_length = config.prompt_length # k
        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        self.context_dim = config.context_dim

        # Ensure data types are consistent (float32 for inputs, bool/int for flags)
        for traj in self.trajectories:
            traj['observations'] = traj['observations'].astype(np.float32)
            traj['actions'] = traj['actions'].astype(np.float32)
            traj['rewards'] = traj['rewards'].astype(np.float32)
            # Contexts are pre-appended in the adapter before dataset initialization
            traj['contexts'] = traj['contexts'].astype(np.float32)
            # Ensure RTG is float32
            if 'rtg' in traj:
                traj['rtg'] = traj['rtg'].astype(np.float32)

        # Calculate statistics for normalization (CPU)
        states = np.concatenate([path['observations'] for path in trajectories], axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        
        # Calculate max return for evaluation monitoring
        self.return_max = np.max([path['rewards'].sum() for path in trajectories])

        # OPTIMIZATION: Pre-process prompt trajectories (Calculate WM errors AND Selection Probabilities)
        # This avoids costly WM inference and probability calculation inside __getitem__.
        print('Preprocessing prompt candidates (Calculating WM errors)...')
        self.processed_prompts = self._preprocess_prompts(prompt_trajectories_list, world_model, config, self.target_device)

        # Create an index map for efficient sequence access
        print('Building dataset index map...')
        self.index_map = self._build_index_map(trajectories)

        # Heuristic for episodes per task (Used to map main trajectory index to the correct prompt pool)
        # We rely on the prompt_trajectories_list structure (List[Tasks] -> List[Prompts])
        if len(self.processed_prompts) > 0:
            # Estimate based on total trajectories divided by number of tasks (length of prompt list)
            self.episodes_per_task = len(self.trajectories) // len(self.processed_prompts)
        else:
            self.episodes_per_task = 1
        
        if self.episodes_per_task == 0:
            self.episodes_per_task = 1

        print(f'Dataset initialized (Lazy Loading). Total sequences available: {len(self.index_map)}. Estimated Episodes/Task: {self.episodes_per_task}')

    # MODIFIED: Calculate and store selection probabilities
    def _preprocess_prompts(self, prompt_list, world_model, config, device):
        processed_prompts = []
        # prompt_list is List[Task] -> List[Prompts]
        for task_prompts in tqdm(prompt_list, desc="Processing Prompts"):
            task_processed = []
            for prompt_traj_raw in task_prompts:
                # Calculate WM errors once per candidate (GPU intensive part)
                prompt_traj = append_error_to_trajectory(
                    world_model, device, self.context_horizon, prompt_traj_raw.copy(),
                    config, self.state_mean, self.state_std
                )
                
                # OPTIMIZATION: Calculate and store selection probabilities during initialization
                selection_data = self._calculate_prompt_selection_probabilities(prompt_traj)
                prompt_traj['selection_data'] = selection_data

                task_processed.append(prompt_traj)
            processed_prompts.append(task_processed)
        return processed_prompts

    # NEW HELPER METHOD: Pre-calculate selection probabilities (CPU intensive)
    def _calculate_prompt_selection_probabilities(self, prompt_traj):
        # Identify available start indices (must have sufficient context history 'h' and length 'k')
        available_indices = np.arange(self.context_horizon, len(prompt_traj['errors']) - self.prompt_length + 1)
        
        if len(available_indices) == 0:
            # Fallback if trajectory is too short: use the end of the trajectory
            fallback_start = max(0, len(prompt_traj['errors']) - self.prompt_length)
            return {'indices': np.array([fallback_start]), 'probs': np.array([1.0])}
        
        # Calculate cumulative error for segments (The computationally intensive part moved here)
        world_model_error = [prompt_traj['errors'][sj : sj + self.prompt_length].sum() for sj in available_indices]
        sum_errors = np.sum(world_model_error)

        # Probabilistic selection based on error magnitude (with robustness checks)
        # Higher error -> higher probability (Self-Guided Prompt)
        if sum_errors > 1e-6 and not np.isnan(sum_errors) and not np.isinf(sum_errors):
            error_probs = np.array(world_model_error) / sum_errors
            # Handle potential numerical instability (NaNs, Infs, or sum not close to 1) by reverting to uniform
            if np.isnan(error_probs).any() or np.isinf(error_probs).any() or abs(error_probs.sum() - 1.0) > 1e-4:
                 error_probs = np.ones_like(available_indices, dtype=float) / len(available_indices)
        else:
            # Uniform selection if errors are negligible or unstable
            error_probs = np.ones_like(available_indices, dtype=float) / len(available_indices)
            
        return {'indices': available_indices, 'probs': error_probs}


    def _build_index_map(self, trajectories):
        index_map = []
        # Map a global index to (trajectory_index, start_time_index)
        for traj_idx, traj in enumerate(trajectories):
            # Every timestep can be the start of a sequence
            for si in range(len(traj['rewards'])):
                index_map.append((traj_idx, si))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        # 1. Map global index to (traj_idx, si)
        traj_idx, si = self.index_map[index]
        traj = self.trajectories[traj_idx]

        # 2. Select and Prepare the Complementary Prompt (On-Demand, CPU)
        # Determine the corresponding task ID pool for the prompt
        task_id_idx = traj_idx // self.episodes_per_task
        
        # Handle edge case where index might exceed the number of processed tasks (due to integer division rounding)
        if task_id_idx >= len(self.processed_prompts):
            task_id_idx = len(self.processed_prompts) - 1
        
        # Select a pre-processed prompt trajectory randomly from the pool
        prompt_candidates = self.processed_prompts[task_id_idx]
        
        # Handle case where a task might have no valid prompts (e.g. if data was missing)
        if not prompt_candidates:
            # Fallback: Create a dummy zero prompt if no candidates exist for the task
            p_s = np.zeros((self.prompt_length, self.state_dim), dtype=np.float32)
            p_a = np.zeros((self.prompt_length, self.act_dim), dtype=np.float32)
            p_r = np.zeros((self.prompt_length, 1), dtype=np.float32)
            p_t = np.zeros(self.prompt_length, dtype=np.int64)
            p_rtg = np.zeros((self.prompt_length + 1, 1), dtype=np.float32)
            # >>> FIX: Include dummy context <<<
            p_c = np.zeros((self.prompt_length, self.context_dim), dtype=np.float32)
        else:
            prompt_traj = random.choice(prompt_candidates)
            # Select segment based on PRE-CALCULATED errors (O(1) operation)
            p_start = self._select_prompt_start_index(prompt_traj)
            # >>> FIX: Updated unpacking (includes p_c) <<<
            p_s, p_a, p_r, p_c, p_t, p_rtg = self._extract_prompt_segment(prompt_traj, p_start)

        # 3. Extract the Main Trajectory Segment (On-Demand, CPU)
        s, a, r, c, d, t, rtg_seg = self._extract_main_segment(traj, si)

        # 4. Padding and Normalization (On-Demand, CPU Tensors)
        
        # Main sequence
        s_padded, c_padded, a_padded, r_padded, d_padded, rtg_padded, t_padded, mask_padded = self._pad_and_normalize(
            s, a, r, c, d, t, rtg_seg, self.horizon
        )

        # Prompt sequence (>>> FIX: Pass p_c instead of None <<<)
        p_s_padded, p_c_padded, p_a_padded, p_r_padded, _, p_rtg_padded, p_t_padded, p_mask_padded = self._pad_and_normalize(
            p_s, p_a, p_r, p_c, None, p_t, p_rtg, self.prompt_length
        )

        # Return CPU tensors. (>>> FIX: Updated return signature - 15 items <<<)
        return (
            s_padded, c_padded, a_padded, r_padded, d_padded, rtg_padded, t_padded, mask_padded,
            p_s_padded, p_c_padded, p_a_padded, p_r_padded, p_rtg_padded, p_t_padded, p_mask_padded
        )

    # Helper methods for lazy loading logic (CPU based)

    # MODIFIED: Use pre-calculated probabilities for O(1) selection
    def _select_prompt_start_index(self, prompt_traj):
        # Retrieve pre-calculated data stored during initialization
        selection_data = prompt_traj['selection_data']
        indices = selection_data['indices']
        probs = selection_data['probs']
        
        # Fast weighted random choice
        selected_index = np.random.choice(indices, p=probs)
        return selected_index

    def _extract_prompt_segment(self, prompt_traj, p_start):
        # Extracts the prompt segment data (CPU)
        p_s = prompt_traj['observations'][p_start:p_start + self.prompt_length]
        p_a = prompt_traj['actions'][p_start:p_start + self.prompt_length]
        p_r = prompt_traj['rewards'][p_start:p_start + self.prompt_length].reshape(-1, 1)
        p_t = np.arange(p_start, p_start + len(p_s))
        
        # >>> FIX: Extract Contexts <<<
        p_c = prompt_traj['contexts'][p_start:p_start + self.prompt_length]
        
        # >>> OPTIMIZATION & FIX: Use pre-calculated RTG and ensure length k+1 <<<
        p_rtg = prompt_traj['rtg'][p_start:p_start + self.prompt_length + 1]
        
        # Ensure RTG has exactly prompt_length + 1 elements
        if p_rtg.shape[0] < self.prompt_length + 1:
             padding_needed = (self.prompt_length + 1) - p_rtg.shape[0]
             # Pad the end if the trajectory ends early
             p_rtg = np.concatenate([p_rtg, np.zeros((padding_needed, 1), dtype=np.float32)], axis=0)
        elif p_rtg.shape[0] > self.prompt_length + 1:
             # Truncate if too long
             p_rtg = p_rtg[:self.prompt_length + 1]
             
        return p_s, p_a, p_r, p_c, p_t, p_rtg

    def _extract_main_segment(self, traj, si):
        # Extracts the main sequence segment data (CPU)
        s = traj['observations'][si : si + self.horizon]
        a = traj['actions'][si : si + self.horizon]
        r = traj['rewards'][si : si + self.horizon].reshape(-1, 1)
        c = traj['contexts'][si : si + self.horizon] # Pre-calculated in Phase 2 data prep
        
        # Handle 'terminals' key robustness
        if 'terminals' in traj:
            d = traj['terminals'][si : si + self.horizon]
        else:
            # Fallback if key is missing (assume not done)
            d = np.zeros(len(s))

        t = np.arange(si, si + len(s))
        t[t >= self.max_episode_steps] = self.max_episode_steps - 1

        # >>> OPTIMIZATION & FIX: Use pre-calculated RTG and ensure length K+1 <<<
        rtg_seg = traj['rtg'][si : si + self.horizon + 1]
        
        # Ensure RTG has exactly horizon + 1 elements
        if rtg_seg.shape[0] < self.horizon + 1:
            padding_needed = (self.horizon + 1) - rtg_seg.shape[0]
            # Pad the end if the trajectory ends early
            rtg_seg = np.concatenate([rtg_seg, np.zeros((padding_needed, 1), dtype=np.float32)], axis=0)
        elif rtg_seg.shape[0] > self.horizon + 1:
            # Truncate if too long
            rtg_seg = rtg_seg[:self.horizon + 1]
            
        return s, a, r, c, d, t, rtg_seg

    def _pad_and_normalize(self, s, a, r, c, d, t, rtg, length):
        # Handles padding, normalization, and conversion to CPU tensors
        tlen = s.shape[0]
        pad_len = length - tlen
        
        # Padding (Numpy, Prefix). Using np.pad is efficient.
        # Data should already be float32 from __init__ / extraction
        s_padded = np.pad(s, ((pad_len, 0), (0, 0)), mode='constant')
        s_padded = (s_padded - self.state_mean) / self.state_std
        
        # Actions padded with -10 (standard DT practice)
        a_padded = np.pad(a, ((pad_len, 0), (0, 0)), mode='constant', constant_values=-10.)
        r_padded = np.pad(r, ((pad_len, 0), (0, 0)), mode='constant')
        
        # RTG has length+1, so pad accordingly
        rtg_len = rtg.shape[0]
        rtg_pad_len = (length + 1) - rtg_len
        rtg_padded = np.pad(rtg, ((rtg_pad_len, 0), (0, 0)), mode='constant') / self.return_scale
        t_padded = np.pad(t, (pad_len, 0), mode='constant')
        mask_padded = np.concatenate([np.zeros(pad_len), np.ones(tlen)], axis=0)

        # Handle optional inputs (c, d)
        if c is not None:
            c_padded = np.pad(c, ((pad_len, 0), (0, 0)), mode='constant')
        else:
            # Placeholder if context is not needed (e.g., for prompt input itself)
            c_padded = np.zeros((length, self.context_dim), dtype=np.float32)

        if d is not None:
            d_padded = np.pad(d, (pad_len, 0), mode='constant', constant_values=2) # Sentinel 2 for padded done flag
        else:
            d_padded = np.ones(length, dtype=np.float32) * 2

        # Convert to CPU Tensors (ensure correct dtypes: float32 for inputs, long for indices/masks)
        # Explicitly cast to float32 to prevent dtype mismatches with AMP
        s_tensor = torch.from_numpy(s_padded).float()
        c_tensor = torch.from_numpy(c_padded).float()
        a_tensor = torch.from_numpy(a_padded).float()
        r_tensor = torch.from_numpy(r_padded).float()
        d_tensor = torch.from_numpy(d_padded).long()
        rtg_tensor = torch.from_numpy(rtg_padded).float()
        t_tensor = torch.from_numpy(t_padded).long()
        mask_tensor = torch.from_numpy(mask_padded).float() # Kept as float for attention mask usage

        return s_tensor, c_tensor, a_tensor, r_tensor, d_tensor, rtg_tensor, t_tensor, mask_tensor