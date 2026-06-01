import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_vals = self.fc3(x)
        return q_vals

class VDNMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values):
        # q_values: (batch_size, n_agents)
        return torch.sum(q_values, dim=-1, keepdim=True)

class QMIXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        self.hyper_w_1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, embed_dim)
        
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, q_values, states):
        b = q_values.shape[0]
        q_values = q_values.view(-1, 1, self.n_agents)
        
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        v = self.V(states).view(-1, 1, 1)
        
        y = torch.bmm(hidden, w_final) + v
        return y.view(b, -1)
