# experiments/baselines/meta_dt_integration/Meta-DT/meta_dt/model.py
import numpy as np
import torch
import torch.nn as nn
import transformers
import logging

# Import the customized GPT2Model from the local trajectory_gpt2 module
# We assume trajectory_gpt2.py is correctly implemented in the same directory.
try:
    # When imported via the adapter, the package structure is recognized
    from meta_dt.trajectory_gpt2 import GPT2Model
except ImportError:
    # Fallback for direct execution or different import contexts
    # We try a relative import as a fallback if the package import fails
    try:
        from .trajectory_gpt2 import GPT2Model
    except ImportError:
        # If both fail, we assume trajectory_gpt2 is directly available in the path
        from trajectory_gpt2 import GPT2Model


logger = logging.getLogger(__name__)

# Renamed from MetaDecisionTransformer (if it was named that way) to DecisionTransformer 
# for consistency with the import in adapter.py.
class DecisionTransformer(nn.Module):
    """
    This model implements the Meta-Decision Transformer architecture.
    It uses a GPT-style transformer to model sequences of (R, z, s, a) tokens.
    It supports conditioning on a prompt sequence (complementary prompt).
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            context_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            prompt_length=5,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.max_length = max_length
        self.prompt_length = prompt_length

        # Configure GPT2 model based on provided arguments (n_layer, n_head, etc. passed via kwargs)
        config = transformers.GPT2Config(
            vocab_size=1,  # Not used in DT
            n_embd=hidden_size,
            **kwargs
        )

        # Initialize the transformer backbone
        self.transformer = GPT2Model(config)

        # Initialize embeddings for different modalities
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        # Meta-DT specific: embedding for the context vector (z_t)
        self.embed_context = torch.nn.Linear(self.context_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # Prediction heads
        # Note: predict_state is often unused in standard DT loss, but kept for completeness.
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, contexts, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):

        # rewards argument is unused in forward pass for standard DT loss, but kept in signature.

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Handle the complementary prompt if provided
        if prompt is not None:
            # The prompt is expected to be a dictionary of processed tensors (prepared by Dataset or _prepare_prompt)
            prompt_states = prompt['states']
            prompt_contexts = prompt['contexts']
            prompt_actions = prompt['actions']
            prompt_returns_to_go = prompt['returns_to_go']
            prompt_timesteps = prompt['timesteps']
            prompt_attention_mask = prompt['attention_mask']
            
            prompt_seq_length = prompt_states.shape[1]

            # Concatenate prompt data with the current trajectory data BEFORE embedding.
            states = torch.cat([prompt_states, states], dim=1)
            contexts = torch.cat([prompt_contexts, contexts], dim=1)
            actions = torch.cat([prompt_actions, actions], dim=1)
            returns_to_go = torch.cat([prompt_returns_to_go, returns_to_go], dim=1)
            timesteps = torch.cat([prompt_timesteps, timesteps], dim=1)

            # Create combined attention mask
            if attention_mask is None:
                # If no mask provided for current traj, assume full visibility
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
            
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            
            # Update seq_length to the combined length
            seq_length = seq_length + prompt_seq_length

        elif attention_mask is None:
            # Default attention mask (1 for visible tokens)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed each modality (now operating on the combined sequence if prompt was used)
        state_embeddings = self.embed_state(states)
        context_embeddings = self.embed_context(contexts)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add positional embeddings (time embeddings)
        state_embeddings = state_embeddings + time_embeddings
        context_embeddings = context_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack inputs in the Meta-DT order: (R_1, z_1, s_1, a_1, R_2, z_2, s_2, a_2, ...)
        # We use 4 modalities consistently.
        stacked_inputs = torch.stack(
            (returns_embeddings, context_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Stack attention mask to match the input shape
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4*seq_length)

        # Pass through the transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # Reshape back to (batch_size, seq_length, 4, hidden_size) and permute
        # Index mapping: 0=R, 1=z, 2=s, 3=a
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # Get predictions based on standard autoregressive DT logic:
        # Predict action a_t based on the state token s_t (index 2), which attends to (R_t, z_t, s_t).
        action_preds = self.predict_action(x[:,2])

        # Predict return/state based on action token a_t (index 3). Often unused in standard DT loss.
        return_preds = self.predict_return(x[:,3])
        state_preds = self.predict_state(x[:,3])
        
        
        # If prompt was used, slice the output to only include predictions for the actual (non-prompt) trajectory
        if prompt is not None:
            # We slice off the first prompt_seq_length predictions, as the sequence was combined.
            action_preds = action_preds[:, prompt_seq_length:]
            return_preds = return_preds[:, prompt_seq_length:]
            state_preds = state_preds[:, prompt_seq_length:]

        return state_preds, action_preds, return_preds

    # MODIFICATION: Signature updated to match integration requirements (config, current_step)
    # This method is used during evaluation (meta_evaluate_episode_rtg_adapted).
    def get_action(self, states, contexts, actions, rewards, returns_to_go, timesteps, prompt, config, current_step, **kwargs):
        
        # Reshape inputs for the model (batch size 1)
        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Handle sequence length truncation (based on DT horizon K=max_length)
        if self.max_length is not None:
            # Truncate to the last K steps
            states = states[:,-self.max_length:]
            contexts = contexts[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # Generate attention mask and padding
            seq_len = states.shape[1]
            padding_len = self.max_length - seq_len
            
            attention_mask = torch.cat([
                torch.zeros(padding_len, device=states.device),
                torch.ones(seq_len, device=states.device)
            ])
            attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
            
            # Pad inputs (pre-padding with zeros)
            def pad_tensor(tensor, dim_size):
                pad_len = self.max_length - tensor.shape[1]
                if pad_len > 0:
                    padding = torch.zeros((tensor.shape[0], pad_len, dim_size), device=tensor.device)
                    return torch.cat([padding, tensor], dim=1)
                return tensor

            states = pad_tensor(states, self.state_dim).to(dtype=torch.float32)
            contexts = pad_tensor(contexts, self.context_dim).to(dtype=torch.float32)
            actions = pad_tensor(actions, self.act_dim).to(dtype=torch.float32)
            returns_to_go = pad_tensor(returns_to_go, 1).to(dtype=torch.float32)

            # Timesteps padding (specific handling as it's long type)
            time_pad_len = self.max_length - timesteps.shape[1]
            if time_pad_len > 0:
                time_padding = torch.zeros((timesteps.shape[0], time_pad_len), device=timesteps.device)
                timesteps = torch.cat([time_padding, timesteps], dim=1).to(dtype=torch.long)
            else:
                timesteps = timesteps.to(dtype=torch.long)

        else:
            attention_mask = None

        # MODIFICATION: Check warm_train_steps to conditionally use the prompt
        # This implements the warm-up phase logic (Meta-DT Algorithm 2).
        # We check if config is provided and if the current step is below the threshold.
        use_prompt = True
        if config and hasattr(config, 'warm_train_steps') and current_step < config.warm_train_steps:
            use_prompt = False

        # Prepare the prompt dictionary if applicable
        prompt_dict = None
        if use_prompt and prompt is not None:
            # Convert the raw prompt trajectory into the format expected by forward()
            prompt_dict = self._prepare_prompt(prompt, states.device)

        # Call the forward pass
        # Note: rewards input to forward is typically unused in DT, hence None is passed.
        _, action_preds, _ = self.forward(
            states, contexts, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, prompt=prompt_dict)

        # Return the action prediction for the last timestep of the main sequence
        return action_preds[0,-1]

    def _prepare_prompt(self, prompt_traj, device):
        """Helper to prepare the prompt trajectory dictionary for the forward pass."""
        
        prompt_length = self.prompt_length
        
        # Convert numpy arrays to tensors
        p_states = torch.from_numpy(prompt_traj['observations']).float().to(device).unsqueeze(0)
        p_actions = torch.from_numpy(prompt_traj['actions']).float().to(device).unsqueeze(0)
        p_rewards = torch.from_numpy(prompt_traj['rewards']).float().to(device).unsqueeze(0)
        p_timesteps = torch.from_numpy(prompt_traj['timesteps']).long().to(device).unsqueeze(0)

        # Calculate Returns-to-Go for the prompt (simple cumulative sum, assuming gamma=1 for prompt utility)
        rtg = torch.zeros_like(p_rewards)
        curr_return = 0
        for t in reversed(range(p_rewards.shape[1])):
            curr_return = p_rewards[0, t] + curr_return
            rtg[0, t] = curr_return
        p_rtg = rtg.unsqueeze(-1)

        # CRITICAL: Ensure contexts are available in the prompt trajectory.
        # This relies on the fix applied in adapter.py (_calculate_wm_errors).
        if 'contexts' not in prompt_traj:
             raise RuntimeError("Prompt trajectory is missing 'contexts'. Ensure Adapter calculates them during evaluation.")

        p_contexts = torch.from_numpy(prompt_traj['contexts']).float().to(device).unsqueeze(0)

        # Handle Padding/Truncation to match configured prompt_length
        actual_len = p_states.shape[1]
        pad_len = prompt_length - actual_len
        
        if pad_len > 0:
            # Pad if shorter (pre-padding)
            p_states = torch.cat([torch.zeros((1, pad_len, self.state_dim), device=device), p_states], dim=1)
            p_contexts = torch.cat([torch.zeros((1, pad_len, self.context_dim), device=device), p_contexts], dim=1)
            p_actions = torch.cat([torch.zeros((1, pad_len, self.act_dim), device=device), p_actions], dim=1)
            p_rtg = torch.cat([torch.zeros((1, pad_len, 1), device=device), p_rtg], dim=1)
            p_timesteps = torch.cat([torch.zeros((1, pad_len), dtype=torch.long, device=device), p_timesteps], dim=1)
            p_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), torch.ones(actual_len, dtype=torch.long, device=device)]).unsqueeze(0)
        else:
            # Truncate if longer (should ideally not happen if selection logic in adapter is correct)
            p_states = p_states[:, -prompt_length:]
            p_contexts = p_contexts[:, -prompt_length:]
            p_actions = p_actions[:, -prompt_length:]
            p_rtg = p_rtg[:, -prompt_length:]
            p_timesteps = p_timesteps[:, -prompt_length:]
            p_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device)

        return {
            'states': p_states,
            'contexts': p_contexts,
            'actions': p_actions,
            'returns_to_go': p_rtg,
            'timesteps': p_timesteps,
            'attention_mask': p_mask
        }