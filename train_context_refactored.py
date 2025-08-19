# experiments/baselines/meta_dt_integration/Meta-DT/train_context_refactored.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from pathlib import Path
import sys

# Ensure imports work correctly when called via the adapter
try:
    from context.model import RNNContextEncoder, RewardDecoder, StateDecoder
    from context.dataset import ContextDataset
except ImportError as e:
    print(f"ERROR: Could not import context modules in train_context_refactored.py. Details: {e}")
    sys.exit(1)

def train_world_model(config, train_data, test_data, save_model_path, log_dir, is_dynamics_env):
    """
    Refactored World Model (WM) training loop (Meta-DT Phase 1).
    MODIFIED: Updated to support efficient parallel data loading for the lazy ContextDataset
    and corrected handling of inputs for the RewardDecoder.
    """
    
    device = config.device
    
    # --- 1. Initialize Models ---
    context_encoder = RNNContextEncoder(
        config.state_dim, config.act_dim, config.context_dim, config.context_hidden_dim
    ).to(device)
    
    if is_dynamics_env:
         dynamics_decoder = StateDecoder(
             config.state_dim, config.act_dim, config.context_dim, config.context_hidden_dim
         ).to(device)
    else:
        # In Meta-DT, the "dynamics_decoder" variable holds the RewardDecoder for reward-based envs
        dynamics_decoder = RewardDecoder(
            config.state_dim, config.act_dim, config.context_dim, config.context_hidden_dim
        ).to(device)

    # --- 2. Initialize Datasets and DataLoaders ---
    
    # Initialize Datasets (now using lazy loading, data remains on CPU)
    train_dataset = ContextDataset(train_data, config.context_horizon, device, is_dynamics_env)
    # Note: In the HPT protocol, test_data is typically the same as train_data.
    test_dataset = ContextDataset(test_data, config.context_horizon, device, is_dynamics_env)

    # Initialize DataLoaders
    # CRITICAL CHANGE: Enable num_workers and pin_memory for efficient parallel CPU loading.
    NUM_WORKERS = 4 # Heuristic: Adjust based on available CPU cores and memory bandwidth.

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.wm_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS, # MODIFIED
        pin_memory=True,         # MODIFIED
        drop_last=True           # Recommended practice to avoid incomplete batches
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.wm_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS, # MODIFIED
        pin_memory=True          # MODIFIED
    )

    # --- 3. Initialize Optimizer ---
    optimizer = optim.Adam(
        list(context_encoder.parameters()) + list(dynamics_decoder.parameters()),
        lr=config.wm_lr
    )
    
    # Calculate number of epochs based on total steps and dataset size
    # Use floor division for steps_per_epoch since drop_last=True
    steps_per_epoch = len(train_dataset) // config.wm_batch_size
    if steps_per_epoch == 0:
        print("ERROR: Batch size is larger than the dataset size or dataset is empty.")
        return
        
    num_epochs = math.ceil(config.wm_train_steps / steps_per_epoch)
    
    print(f"Starting WM training. Target Steps: {config.wm_train_steps}. Calculated Epochs: {num_epochs}.")

    # --- 4. Training Loop ---
    metrics_log = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        
        # --- Validation Phase (Start of Epoch) ---
        context_encoder.eval()
        dynamics_decoder.eval()
        
        val_loss_total = 0.0
        val_reward_loss_total = 0.0
        val_state_loss_total = 0.0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                
                # CRITICAL CHANGE: Unpack data including the actual_next_state (s_t+1).
                (state_seg, action_seg, reward_seg, 
                 next_state_seg, next_action_seg, next_reward_seg, 
                 target, actual_next_state) = data

                # Move data to GPU (non_blocking=True for optimized transfer)
                state_seg = state_seg.to(device, non_blocking=True)
                action_seg = action_seg.to(device, non_blocking=True)
                reward_seg = reward_seg.to(device, non_blocking=True)
                next_state_seg = next_state_seg.to(device, non_blocking=True)
                next_action_seg = next_action_seg.to(device, non_blocking=True)
                next_reward_seg = next_reward_seg.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                actual_next_state = actual_next_state.to(device, non_blocking=True)

                # Calculate context using the 'current' history (up to t-1)
                # GRU input shape: [sequence_len (H), batch_size, dim]
                contexts = context_encoder(
                    state_seg.transpose(0, 1),
                    action_seg.transpose(0, 1),
                    reward_seg.transpose(0, 1)
                )

                # Prepare inputs for the decoder (at time t)
                # We extract the last element of the 'next' segments (which corresponds to time t)
                current_states = next_state_seg[:, -1, :]
                current_actions = next_action_seg[:, -1, :]
                current_rewards = next_reward_seg[:, -1, :]

                # Prediction
                if is_dynamics_env:
                    # StateDecoder signature: (s_t, a_t, r_t, s_t+1, context)
                    # We use actual_next_state for the s_t+1 input.
                    prediction = dynamics_decoder(current_states, current_actions, current_rewards, actual_next_state, contexts)
                    loss = F.mse_loss(prediction, target)
                    val_state_loss_total += loss.item()
                else:
                    # RewardDecoder signature: (s_t, a_t, s_t+1, context)
                    # We use actual_next_state for the s_t+1 input.
                    prediction = dynamics_decoder(current_states, current_actions, actual_next_state, contexts)
                    loss = F.mse_loss(prediction, target)
                    val_reward_loss_total += loss.item()
                
                val_loss_total += loss.item()

        # Calculate average validation losses
        # Handle potential division by zero if test_loader is empty
        num_test_batches = len(test_loader)
        if num_test_batches == 0:
            avg_val_loss = 0
            avg_val_reward_loss = 0
            avg_val_state_loss = 0
        else:
            avg_val_loss = val_loss_total / num_test_batches
            avg_val_reward_loss = val_reward_loss_total / num_test_batches
            avg_val_state_loss = val_state_loss_total / num_test_batches

        # --- Training Phase ---
        context_encoder.train()
        dynamics_decoder.train()
        
        train_loss_total = 0.0
        
        # Use tqdm for progress tracking within the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, data in enumerate(pbar):
            
            # CRITICAL CHANGE: Unpack and move data to the GPU
            (state_seg, action_seg, reward_seg, 
             next_state_seg, next_action_seg, next_reward_seg, 
             target, actual_next_state) = data

            state_seg = state_seg.to(device, non_blocking=True)
            action_seg = action_seg.to(device, non_blocking=True)
            reward_seg = reward_seg.to(device, non_blocking=True)
            next_state_seg = next_state_seg.to(device, non_blocking=True)
            # next_action_seg and next_reward_seg are only used to extract the last element
            target = target.to(device, non_blocking=True)
            actual_next_state = actual_next_state.to(device, non_blocking=True)

            # Calculate context (same logic as validation)
            contexts = context_encoder(
                state_seg.transpose(0, 1),
                action_seg.transpose(0, 1),
                reward_seg.transpose(0, 1)
            )

            # Prepare inputs for the decoder (same logic as validation)
            current_states = next_state_seg[:, -1, :]
            # Extracting these on the GPU directly
            current_actions = next_action_seg[:, -1, :].to(device, non_blocking=True)
            current_rewards = next_reward_seg[:, -1, :].to(device, non_blocking=True)

            # Prediction and Loss
            if is_dynamics_env:
                prediction = dynamics_decoder(current_states, current_actions, current_rewards, actual_next_state, contexts)
            else:
                prediction = dynamics_decoder(current_states, current_actions, actual_next_state, contexts)
                
            loss = F.mse_loss(prediction, target)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            # Update progress bar with current loss
            pbar.set_postfix({'Train Loss': loss.item(), 'Val Loss': avg_val_loss})

        avg_train_loss = train_loss_total / len(train_loader)

        # --- Logging and Saving ---
        
        # Calculate current step (approximation based on batches processed)
        current_step = (epoch + 1) * len(train_loader)
        
        metrics = {
            'epoch': epoch + 1,
            'step': current_step,
            'train_loss': avg_train_loss,
            'val_total_loss': avg_val_loss,
            # Include specific losses for validation plots (Requirement 4b)
            'val_reward_loss': avg_val_reward_loss if not is_dynamics_env else 0,
            'val_state_loss': avg_val_state_loss if is_dynamics_env else 0,
        }
        metrics_log.append(metrics)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss and num_test_batches > 0:
            best_val_loss = avg_val_loss
            torch.save({
                'context_encoder': context_encoder.state_dict(),
                'dynamics_decoder': dynamics_decoder.state_dict(),
                'epoch': epoch + 1,
                'config': config, # Save config for provenance
            }, save_model_path)

    # Save metrics CSV (Required for HPT analysis)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame(metrics_log)
    
    # Ensure required columns exist even if they were zeroed out (e.g. reward loss for dynamics env)
    if 'val_reward_loss' not in df_metrics.columns:
        df_metrics['val_reward_loss'] = 0
    if 'val_state_loss' not in df_metrics.columns:
        df_metrics['val_state_loss'] = 0
        
    df_metrics.to_csv(log_dir_path / 'metrics.csv', index=False)
    print(f"WM Training finished. Best Val Loss: {best_val_loss:.6f}. Metrics saved to {log_dir_path / 'metrics.csv'}")