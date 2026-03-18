"""
training/buffer.py  --  PPO élmény buffer GAE számítással
"""
import torch
import numpy as np

class PPOBuffer:
    def __init__(self): self.reset()

    def reset(self):
        self.states=[]; self.legal_actions=[]; self.actions=[]
        self.log_probs=[]; self.values=[]; self.rewards=[]; self.episode_ends=[]

    def add(self, state, legal_actions, action, log_prob, value, reward, episode_end):
        self.states.append(state.squeeze(0).detach().cpu())
        self.legal_actions.append(legal_actions)
        self.actions.append(action.detach().cpu().squeeze())
        self.log_probs.append(log_prob.detach().cpu().squeeze())
        self.values.append(value.detach().cpu().squeeze().float())
        self.rewards.append(float(reward))
        self.episode_ends.append(bool(episode_end))

    def compute_gae(self, gamma=0.99, lam=0.95):
        n = len(self.rewards)
        if n == 0: return torch.zeros(0), torch.zeros(0)
        rewards    = torch.tensor(self.rewards,      dtype=torch.float32)
        dones      = torch.tensor(self.episode_ends, dtype=torch.float32)
        values     = torch.stack(self.values).float()
        advantages = torch.zeros(n, dtype=torch.float32)
        last_gae   = 0.0
        for t in reversed(range(n)):
            not_done = 1.0 - float(dones[t])
            nv       = float(values[t + 1]) if t + 1 < n else 0.0
            delta    = float(rewards[t]) + gamma * nv * not_done - float(values[t])
            last_gae = delta + gamma * lam * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def __len__(self): return len(self.states)
