# experiments/baselines/meta_dt_integration/Meta-DT/context/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
# Removed unused imports (pickle, os)

# MODIFICATION: This class implements lazy loading (on-demand segmentation) on the CPU
# to resolve the memory explosion issue caused by eager preprocessing.

class ContextDataset(Dataset):
    
    def __init__(self, data_dict, horizon, device, is_dynamics_env=True):
        """
        Args:
            data_dict (dict): Pooled dataset dictionary (from load_multi_task_data).
            horizon (int): The context horizon (h) for the RNN.
            device (torch.device): Target device (data will be moved here during training).
            is_dynamics_env (bool): True if predicting next_state, False if predicting reward.
        """
        self.horizon = horizon
        # Device is kept for reference, but data stays on CPU initially.
        self.target_device = device 
        self.is_dynamics_env = is_dynamics_env

        # Store data as numpy arrays (on CPU RAM)
        # Ensure inputs are float32 immediately for PyTorch compatibility
        self.states = data_dict['observations'].astype(np.float32)
        self.actions = data_dict['actions'].astype(np.float32)
        self.rewards = data_dict['rewards'].reshape(-1, 1).astype(np.float32)
        self.next_states = data_dict['next_observations'].astype(np.float32)
        
        # Identify episode boundaries (terminals OR timeouts)
        terminals = data_dict['terminals'].astype(bool).flatten()
        timeouts = data_dict['timeouts'].astype(bool).flatten()
        episode_ends = np.logical_or(terminals, timeouts)

        # CRITICAL: Pre-calculate the start index of the episode for each transition
        # This allows efficient boundary checking during on-demand segmentation.
        self.episode_start_indices = self._calculate_episode_starts(episode_ends)

        # The original memory-intensive method (parse_trajectory_segment) is removed.

    def _calculate_episode_starts(self, episode_ends):
        """Calculates the start index of the episode for every transition."""
        start_indices = np.zeros(len(self.states), dtype=int)
        current_start = 0
        for i in range(len(self.states)):
            start_indices[i] = current_start
            # Check if the episode ends at the current index (safe boundary check)
            if i < len(episode_ends) and episode_ends[i]:
                current_start = i + 1
        return start_indices

    def __len__(self):
        return self.states.shape[0]

    def _get_segment(self, idx, segment_type='current'):
        """Helper to extract and pad a segment on-demand (CPU)."""
        H = self.horizon
        initial_state_idx = self.episode_start_indices[idx]

        # Determine the start and end indices for the slice
        if segment_type == 'current':
            # Context: (t-H ... t-1)
            end_idx = idx
            start_idx = max(initial_state_idx, idx - H)
        elif segment_type == 'next':
            # Context: (t-H+1 ... t). Used for teacher forcing compatibility.
            end_idx = idx + 1
            start_idx = max(initial_state_idx, idx - H + 1)
        else:
            raise ValueError("Invalid segment type")

        # Extract the segment from raw data (NumPy)
        if initial_state_idx >= end_idx:
             # Slice is empty (e.g., start of episode)
            state_seg = np.zeros((0, self.states.shape[1]), dtype=np.float32)
            action_seg = np.zeros((0, self.actions.shape[1]), dtype=np.float32)
            reward_seg = np.zeros((0, self.rewards.shape[1]), dtype=np.float32)
        else:
            # Data is already float32 from __init__
            state_seg = self.states[start_idx : end_idx]
            action_seg = self.actions[start_idx : end_idx]
            reward_seg = self.rewards[start_idx : end_idx]

        # Pre-padding if the segment is shorter than the horizon
        length_gap = H - state_seg.shape[0]
        
        if length_gap > 0:
            state_seg = np.pad(state_seg, ((length_gap, 0), (0, 0)), mode='constant')
            action_seg = np.pad(action_seg, ((length_gap, 0), (0, 0)), mode='constant')
            reward_seg = np.pad(reward_seg, ((length_gap, 0), (0, 0)), mode='constant')

        # Convert to CPU tensors. The DataLoader handles batching and transfer to GPU.
        return (
            torch.from_numpy(state_seg),
            torch.from_numpy(action_seg),
            torch.from_numpy(reward_seg)
        )

    def __getitem__(self, idx):
        
        # 1. Get the current context segment (ends at t-1)
        state_seg, action_seg, reward_seg = self._get_segment(idx, 'current')

        # 2. Get the next context segment (ends at t)
        next_state_seg, next_action_seg, next_reward_seg = self._get_segment(idx, 'next')

        # 3. Determine the target AND the actual next state (s_t+1)
        # FIX: We must always return s_t+1 because the RewardDecoder requires it as input.
        actual_next_state = self.next_states[idx]

        if self.is_dynamics_env:
            # Target is s_{t+1}
            target = actual_next_state
        else:
            # Target is r_t
            target = self.rewards[idx]
            if target.ndim > 1:
                target = target.flatten()

        # Convert targets to CPU tensors (already float32)
        target_tensor = torch.from_numpy(target)
        next_state_tensor = torch.from_numpy(actual_next_state)

        # Return the full set of inputs required by the trainer
        # NEW RETURN SIGNATURE: Added next_state_tensor (8 items total)
        return (
            state_seg, action_seg, reward_seg, 
            next_state_seg, next_action_seg, next_reward_seg, 
            target_tensor,
            next_state_tensor # Added s_t+1
        )