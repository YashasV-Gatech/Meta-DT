# experiments/baselines/meta_dt_integration/Meta-DT/train_meta_dt_refactored.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
from torch.utils.data import DataLoader
from pathlib import Path
import sys
# Import AMP utilities
from torch.amp import GradScaler, autocast

# Import necessary components
# Assuming these imports work correctly based on the adapter's sys.path modifications
try:
    from meta_dt.model import DecisionTransformer
    from meta_dt.dataset_adapted import MetaDT_Dataset_Adapted
    from meta_dt.evaluation_adapted import meta_evaluate_episode_rtg_adapted
    # Trainer is not strictly needed as the loop is implemented here
    # from meta_dt.trainer import MetaDT_Trainer
except ImportError as e:
    print(f"ERROR: Could not import Meta-DT modules in train_meta_dt_refactored.py. Details: {e}")
    # Handle error appropriately if imports fail
    DecisionTransformer = None
    MetaDT_Dataset_Adapted = None
    meta_evaluate_episode_rtg_adapted = None


# Helper function to append contexts and timesteps (needed for evaluation prompts)
def _append_context_to_trajectory(traj, context_encoder, horizon, device):
    """Helper to append contexts and timesteps to a trajectory for evaluation prompts."""
    context_encoder.eval()
    
    # Ensure inputs are float32
    states = traj['observations'].astype(np.float32)
    actions = traj['actions'].astype(np.float32)
    rewards = traj['rewards'].reshape(-1, 1).astype(np.float32)

    # 1. Segmentation and Padding (CPU)
    states_segment, actions_segment, rewards_segment = [], [], []
    
    for idx in range(len(states)):
        start_idx = max(0, idx - horizon)
        end_idx = idx # Context up to t-1

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

    # 2. Inference (GPU)
    # Tensors are already float32 from numpy conversion
    states_segment_t = torch.from_numpy(np.stack(states_segment, axis=0)).to(device)
    actions_segment_t = torch.from_numpy(np.stack(actions_segment, axis=0)).to(device)
    rewards_segment_t = torch.from_numpy(np.stack(rewards_segment, axis=0)).to(device)

    with torch.no_grad():
        # Encoder expects input shape [sequence_len (H), batch_size, dim]
        contexts = context_encoder(
            states_segment_t.transpose(0, 1),
            actions_segment_t.transpose(0, 1),
            rewards_segment_t.transpose(0, 1)
        )
    
    traj['contexts'] = contexts.detach().cpu().numpy()

    # CRITICAL FIX: Ensure timesteps are present, required by the model's _prepare_prompt method
    if 'timesteps' not in traj:
        traj['timesteps'] = np.arange(len(states))
        
    return traj


def train_decision_transformer(
    config,
    train_trajectories,
    context_encoder,
    dynamics_decoder,
    prompt_trajectories_list,
    get_env_fn,
    save_model_path,
    log_dir,
    eval_task_ids,
    is_dynamics_env
):
    """
    Trains the Decision Transformer (Phase 2).
    """
    if MetaDT_Dataset_Adapted is None or DecisionTransformer is None:
        print("ERROR: Required classes not imported. Cannot train DT.")
        return

    print("\n--- Executing DT Training (Refactored with DataLoader) ---")
    print(f"  Steps: {config.dt_train_steps}, Batch Size: {config.dt_batch_size}, LR: {config.dt_lr}")

    # 1. Initialize Dataset (Handles normalization calculation and prompt preprocessing)
    train_dataset = MetaDT_Dataset_Adapted(
        trajectories=train_trajectories,
        horizon=config.dt_horizon,
        max_episode_steps=config.max_episode_steps,
        return_scale=config.return_scale,
        device=config.device,
        prompt_trajectories_list=prompt_trajectories_list,
        config=config,
        world_model=(context_encoder, dynamics_decoder)
    )

    # 2. Initialize DataLoader
    # Optimize DataLoader settings
    NUM_WORKERS = 8 # Increase workers now that CPU load per worker is reduced
    
    # Set persistent_workers=True if dataset is large enough and OS supports it, to avoid worker respawn overhead
    use_persistent = True if NUM_WORKERS > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dt_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=True, # Drop last incomplete batch
        persistent_workers=use_persistent
    )

    # 3. Initialize Model
    model = DecisionTransformer(
        state_dim=config.state_dim,
        act_dim=config.act_dim,
        max_length=config.dt_horizon,
        max_ep_len=config.max_episode_steps,
        hidden_size=config.dt_embed_dim,
        n_layer=config.dt_n_layer,
        n_head=config.dt_n_head,
        n_inner=4*config.dt_embed_dim,
        activation_function=config.dt_activation_function,
        n_positions=1024,
        resid_pdrop=config.dt_dropout,
        attn_pdrop=config.dt_dropout,
        context_dim=config.context_dim,
        prompt_length=config.prompt_length
    ).to(config.device)

    # 4. Initialize Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.dt_lr,
        weight_decay=config.dt_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/config.dt_warmup_steps, 1)
    )

    # Initialize GradScaler for AMP
    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    # 5. Training Loop
    start_time = time.time()
    metrics_log = []
    pbar = tqdm(range(config.dt_train_steps), desc="DT Training")
    data_iter = iter(train_loader)

    for current_step in pbar:
        
        # Fetch batch
        try:
            batch = next(data_iter)
        except StopIteration:
            # Re-initialize iterator if epoch finishes
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Move data to the target device (use non_blocking=True when pin_memory=True for optimized transfer)
        batch = [item.to(config.device, non_blocking=True) for item in batch]

        # >>> UPDATE: Unpack 15 items <<<
        (
            states, contexts, actions, rewards, dones, rtg, timesteps, masks,
            p_states, p_contexts, p_actions, p_rewards, p_rtg, p_timesteps, p_masks
        ) = batch

        # Prepare Prompt Dictionary (Must match model.forward expectations)
        prompt_dict = {
            'states': p_states,
            'contexts': p_contexts, # >>> FIX: Include prompt contexts <<<
            'actions': p_actions,
            # Input RTG to the model should be length K (slice off the last element)
            'returns_to_go': p_rtg[:, :-1], 
            'timesteps': p_timesteps,
            'attention_mask': p_masks
        }

        # Determine if prompt should be used based on warm-up steps
        use_prompt = prompt_dict if current_step >= config.warm_train_steps else None

        optimizer.zero_grad()

        # >>> AMP: Use autocast for the forward pass <<<
        with autocast(device_type='cuda', enabled=use_amp):
            # Forward Pass (Input RTG should be length K)
            model.train()
            action_target = torch.clone(actions)
            
            state_preds, action_preds, return_preds = model.forward(
                states, contexts, actions, None, rtg[:, :-1], timesteps, attention_mask=masks, prompt=use_prompt
            )

            # Calculate Loss (MSE on actions)
            act_dim = action_preds.shape[2]
            # Only calculate loss on valid (non-padded) tokens
            valid_tokens_mask = masks.reshape(-1) > 0
            action_preds_flat = action_preds.reshape(-1, act_dim)[valid_tokens_mask]
            action_target_flat = action_target.reshape(-1, act_dim)[valid_tokens_mask]
            
            loss = F.mse_loss(action_preds_flat, action_target_flat)

        # >>> AMP: Use scaler for the backward pass and optimization <<<
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Unscale before clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{current_lr:.1e}")
        metrics_log.append({'step': current_step + 1, 'train_action_loss': loss.item(), 'lr': current_lr})

        # --- Evaluation Checkpoint ---
        # Check (current_step + 1) for alignment with interval
        if (current_step + 1) % config.dt_eval_interval == 0:
            print(f"\nEvaluating at step {current_step + 1}...")
            eval_returns = {}
            
            # Evaluate on a subset of tasks
            for eval_task_id in eval_task_ids:
                
                # Ensure the eval_task_id corresponds to a valid index in the prompt list
                # Assuming eval_task_ids align with the indices used to create prompt_trajectories_list
                try:
                    # In the provided setup (adapter.py), prompt_trajectories_list is built sequentially 
                    # over TRAIN_TASK_IDS. So index access should match the task_id if TRAIN_TASK_IDS starts at 0.
                    prompt_candidates = prompt_trajectories_list[eval_task_id]
                except IndexError:
                    print(f"WARNING: eval_task_id {eval_task_id} out of bounds for prompt list. Skipping evaluation for this task.")
                    continue

                if not prompt_candidates:
                    print(f"WARNING: No prompt candidates for task {eval_task_id}. Skipping evaluation for this task.")
                    continue

                env = get_env_fn(eval_task_id)
                
                # Select a prompt for the evaluation task (using high-return trajectory)
                # Use the best trajectory as the prompt during intermediate eval
                prompt_raw = max(prompt_candidates, key=lambda x: x['rewards'].sum())

                # CRITICAL FIX: Append contexts and timesteps to the prompt trajectory
                # We must copy the raw prompt so we don't modify the original data structure
                prompt = _append_context_to_trajectory(
                    prompt_raw.copy(), context_encoder, config.context_horizon, config.device
                )

                # CRITICAL FIX: Use keyword arguments and correct 'prompt'. Unpack 3 return values.
                returns, _, _ = meta_evaluate_episode_rtg_adapted(
                    env=env,
                    config=config,
                    model=model,
                    context_encoder=context_encoder,
                    state_mean=train_dataset.state_mean,
                    state_std=train_dataset.state_std,
                    # Use the max return seen in training data as the target return for evaluation
                    target_return=train_dataset.return_max / config.return_scale,
                    prompt=prompt,          # FIX: Changed from prompt_trajectory
                    device=config.device,
                    current_step=current_step+1 # Pass current step for warm-up logic
                )
                eval_returns[eval_task_id] = returns
                env.close()

            if eval_returns:
                avg_eval_return = np.mean(list(eval_returns.values()))
                print(f"  Average Eval Return: {avg_eval_return:.4f}")
                # Log evaluation metrics
                metrics_log.append({'step': current_step + 1, 'eval_return': avg_eval_return})
            else:
                print("  Evaluation skipped due to missing data.")

    # 6. Save Model and Logs
    print(f"Training finished. Total time: {(time.time() - start_time)/60:.2f} minutes.")

    # Save the model state dict AND normalization statistics
    print(f"Saving Decision Transformer model and normalization stats to {save_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        # Ensure stats are saved as numpy arrays for compatibility
        'state_mean': np.array(train_dataset.state_mean),
        'state_std': np.array(train_dataset.state_std),
    }, save_model_path)
    
    # Save metrics log
    if log_dir:
        df_metrics = pd.DataFrame(metrics_log)
        # Fill NaN values for cleaner CSV (e.g. eval_return is NaN during training steps)
        # Use ffill to carry forward the last valid metric value if desired, or fillna(0)
        # We will use forward fill for training loss/lr, and keep NaNs for eval return initially
        df_metrics['train_action_loss'] = df_metrics['train_action_loss'].ffill()
        df_metrics['lr'] = df_metrics['lr'].ffill()
        
        log_path = Path(log_dir) / 'metrics.csv'
        df_metrics.to_csv(log_path, index=False)
        print(f"Training metrics saved to {log_path}")