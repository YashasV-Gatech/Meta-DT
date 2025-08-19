# experiments/baselines/meta_dt_integration/Meta-DT/context/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        # Using Xavier uniform initialization for consistency
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# GeneralEncoder is typically unused in the standard Meta-DT flow but retained if needed.
class GeneralEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(GeneralEncoder, self).__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())
        self.next_state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())

        self.gru = nn.GRU(input_size=3*context_dim, hidden_size=context_hidden_dim, num_layers=1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param) # Orthogonal init for RNNs

        # output layer, output z
        self.context_output = nn.Linear(context_hidden_dim, context_dim)
        self.apply(weights_init_)

    def forward(self, states, actions, rewards, next_states):
        """
        Inputs should be given in form [sequence_len (H), batch_size, dim].
        """
        states = states.reshape((-1, *states.shape[-2:]))
        actions = actions.reshape((-1, *actions.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        next_states = next_states.reshape((-1, *next_states.shape[-2:]))

        # extract features
        hs = self.state_encoder(states)
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        # hn_s is calculated but unused in the concatenation below in this variant
        hn_s = self.next_state_encoder(next_states)

        h = torch.cat((ha, hs, hr), dim=-1)

        # gru_output: [seq_len, batch_size, hidden_dim]
        gru_output, _ = self.gru(h)
        contexts = self.context_output(gru_output[-1])
        return contexts

class RNNContextEncoder(nn.Module):
    """
    The primary Context Encoder (E_psi) implementation using a GRU.
    Architecture standardized as per Appendix B.7.4.
    """
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(RNNContextEncoder, self).__init__()
        # Input feature extractors (1-layer MLPs)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())

        # GRU layer
        self.gru = nn.GRU(input_size=3*context_dim, hidden_size=context_hidden_dim, num_layers=1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # Output layer
        self.context_output = nn.Linear(context_hidden_dim, context_dim)
        self.apply(weights_init_)

    def forward(self, states, actions, rewards):
        """
        Input shapes expected: [sequence_len (H), batch_size, dim].
        """
        # Ensure inputs are correctly shaped
        states = states.reshape((-1, *states.shape[-2:]))
        actions = actions.reshape((-1, *actions.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        # Embed inputs
        hs = self.state_encoder(states)
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=-1)

        # Process sequence
        gru_output, _ = self.gru(h)

        # Output the context using the last hidden state
        contexts = self.context_output(gru_output[-1])
        return contexts

class RewardDecoder(nn.Module):
    """
    Reward Decoder implementation.
    MODIFICATION: Standardized to a 2-layer MLP (Input -> Hidden -> Output).
    """
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(RewardDecoder, self).__init__()
        # Input feature extractors
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())

        # 2-Layer MLP Decoder
        # Input dim = state_feat + action_feat + next_state_feat + context(z) = 4 * context_dim
        self.linear1 = nn.Linear(context_dim*4, context_hidden_dim)
        # Removed intermediate layer (self.linear2)
        self.linear_out = nn.Linear(context_hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action, next_state, context):
        # Extract features
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        hs_next = self.state_encoder(next_state)

        # Concatenate inputs
        h = torch.cat((hs, ha, hs_next, context), dim=-1)

        # Process through 2-layer MLP
        h = F.relu(self.linear1(h))
        # Removed intermediate activation
        reward_predict = self.linear_out(h)

        return reward_predict

class StateDecoder(nn.Module):
    """
    State Transition Decoder implementation.
    MODIFICATION: Standardized to a 2-layer MLP (Input -> Hidden -> Output).
    """
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(StateDecoder, self).__init__()
        # Input feature extractors
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        # Reward encoder is defined but unused in the forward pass in the original implementation.
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())

        # 2-Layer MLP Decoder
        # Input dim = state_feat + action_feat + context(z) = 3 * context_dim
        self.linear1 = nn.Linear(context_dim*3, context_hidden_dim)
        # Removed intermediate layer (self.linear2)
        self.linear_out = nn.Linear(context_hidden_dim, state_dim)
        self.apply(weights_init_)

    # The signature includes reward and next_state for compatibility with the unified trainer,
    # even if they are not used in the concatenation logic (adhering to original implementation).
    def forward(self, state, action, reward, next_state, context):
        # Extract features
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)

        # Concatenate inputs
        h = torch.cat((hs, ha, context), dim=-1)

        # Process through 2-layer MLP
        h = F.relu(self.linear1(h))
        # Removed intermediate activation
        state_predict = self.linear_out(h)

        return state_predict