import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
from networks import QNetwork, VDNMixer, QMIXMixer

class GlobalReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition):
        # transition: {'obs': dict, 'actions': dict, 'rewards': dict, 'next_obs': dict, 'dones': dict}
        self.buffer.append(transition)
            
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)

class MARLTrainer:
    """Treinador centralizado unificado para IQL, VDN e QMIX."""
    def __init__(self, agents_list, obs_dim, action_dim, algo="IQL", lr=5e-4, gamma=0.99, use_norm=False, norm_factor=100.0, use_clip=False):
        self.agents_list = agents_list
        self.n_agents = len(agents_list)
        self.algo = algo
        self.gamma = gamma
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.use_norm = use_norm
        self.norm_factor = norm_factor
        self.use_clip = use_clip
        
        # Redes individuais para cada agente
        self.q_nets = nn.ModuleDict({
            agent: QNetwork(obs_dim, action_dim) for agent in agents_list
        })
        self.target_q_nets = nn.ModuleDict({
            agent: QNetwork(obs_dim, action_dim) for agent in agents_list
        })
        self.target_q_nets.load_state_dict(self.q_nets.state_dict())
        
        self.params = list(self.q_nets.parameters())
        
        self.mixer = None
        self.target_mixer = None
        state_dim = obs_dim * self.n_agents # Global state artificial via concatenação
        
        if algo == "VDN":
            self.mixer = VDNMixer()
            self.target_mixer = VDNMixer()
            self.params += list(self.mixer.parameters())
        elif algo == "QMIX":
            self.mixer = QMIXMixer(self.n_agents, state_dim)
            self.target_mixer = QMIXMixer(self.n_agents, state_dim)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.params += list(self.mixer.parameters())
            
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        
    def get_actions(self, obs_dict, epsilon=0.1):
        if self.algo == "Aleatório":
            epsilon = 1.0  # Força exploração total
            
        actions = {}
        for agent in self.agents_list:
            if agent not in obs_dict:
                actions[agent] = 0
                continue
                
            if np.random.rand() < epsilon:
                actions[agent] = np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_dict[agent]).unsqueeze(0)
                    q_values = self.q_nets[agent](obs_t)
                    actions[agent] = q_values.argmax(dim=-1).item()
        return actions
        
    def _get_q_values(self, obs_batch, acts_batch, use_target=False):
        q_vals = []
        for i, agent in enumerate(self.agents_list):
            obs = obs_batch[:, i, :]
            acts = acts_batch[:, i].unsqueeze(-1)
            net = self.target_q_nets[agent] if use_target else self.q_nets[agent]
            q = net(obs).gather(1, acts)
            q_vals.append(q)
        return torch.cat(q_vals, dim=-1) 
        
    def _get_max_q_values(self, next_obs_batch, use_target=True):
        max_q_vals = []
        for i, agent in enumerate(self.agents_list):
            obs = next_obs_batch[:, i, :]
            net = self.target_q_nets[agent] if use_target else self.q_nets[agent]
            max_q = net(obs).max(dim=1)[0].unsqueeze(-1)
            max_q_vals.append(max_q)
        return torch.cat(max_q_vals, dim=-1)
        
    def update(self, batch):
        if self.algo == "Aleatório":
            return 0.0
            
        b_size = len(batch)
        
        obs = torch.FloatTensor(np.array([[b['obs'][ag] for ag in self.agents_list] for b in batch]))
        acts = torch.LongTensor(np.array([[b['actions'][ag] for ag in self.agents_list] for b in batch]))
        rews = torch.FloatTensor(np.array([[b['rewards'][ag] for ag in self.agents_list] for b in batch]))
        next_obs = torch.FloatTensor(np.array([[b['next_obs'][ag] for ag in self.agents_list] for b in batch]))
        dones = torch.FloatTensor(np.array([[b['dones'][ag] for ag in self.agents_list] for b in batch]))
        
        if self.use_norm:
            rews = rews / self.norm_factor
        
        states = obs.view(b_size, -1)
        next_states = next_obs.view(b_size, -1)
        
        q_values = self._get_q_values(obs, acts)
        with torch.no_grad():
            max_next_q = self._get_max_q_values(next_obs, use_target=True)
            
        if self.algo == "IQL":
            target = rews + self.gamma * (1 - dones) * max_next_q
            loss = F.mse_loss(q_values, target)
        else:
            global_rew = rews.sum(dim=-1, keepdim=True)
            global_done = dones.max(dim=-1, keepdim=True)[0]
            
            q_tot = self.mixer(q_values, states) if self.algo == "QMIX" else self.mixer(q_values)
            next_q_tot = self.target_mixer(max_next_q, next_states) if self.algo == "QMIX" else self.target_mixer(max_next_q)
            
            target = global_rew + self.gamma * (1 - global_done) * next_q_tot
            loss = F.mse_loss(q_tot, target)
            
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.use_clip:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            
        self.optimizer.step()
        return loss.item()
        
    def update_target(self):
        self.target_q_nets.load_state_dict(self.q_nets.state_dict())
        if self.algo in ["VDN", "QMIX"]:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path):
        data = {
            'q_nets': self.q_nets.state_dict(),
            'target_q_nets': self.target_q_nets.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.mixer is not None:
            data['mixer'] = self.mixer.state_dict()
            data['target_mixer'] = self.target_mixer.state_dict()
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path)
        self.q_nets.load_state_dict(data['q_nets'])
        self.target_q_nets.load_state_dict(data['target_q_nets'])
        self.optimizer.load_state_dict(data['optimizer'])
        if self.mixer is not None:
            self.mixer.load_state_dict(data['mixer'])
            self.target_mixer.load_state_dict(data['target_mixer'])
