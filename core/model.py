"""
core/model.py  --  AdvancedPokerAI (v4)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, expansion=2, dropout=0.05):
        super().__init__()
        inner=hidden_size*expansion
        self.norm=nn.LayerNorm(hidden_size); self.fc1=nn.Linear(hidden_size,inner)
        self.fc2=nn.Linear(inner,hidden_size); self.drop=nn.Dropout(dropout)
    def forward(self, x):
        r=x; x=self.norm(x); x=F.gelu(self.fc1(x)); x=self.drop(self.fc2(x)); return x+r

class TemporalBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.05):
        super().__init__()
        self.norm=nn.LayerNorm(hidden_size); self.fc1=nn.Linear(hidden_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size); self.drop=nn.Dropout(dropout)
    def forward(self, x):
        r=x; x=self.norm(x); x=F.gelu(self.fc1(x)); x=self.drop(self.fc2(x)); return x+r

class AdvancedPokerAI(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super().__init__()
        self.hidden_size=hidden_size; self.action_size=action_size
        self.input_norm=nn.LayerNorm(state_size)
        self.input_proj=nn.Linear(state_size, hidden_size)
        self.res_blocks=nn.ModuleList([ResidualBlock(hidden_size,2,0.05) for _ in range(4)])
        self.temporal=nn.ModuleList([TemporalBlock(hidden_size,0.05) for _ in range(2)])
        self.actor_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size//2),nn.GELU(),nn.Linear(hidden_size//2,action_size))
        self.critic_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size//2),nn.GELU(),nn.Linear(hidden_size//2,1))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim()>=2 and 'weight' in name: nn.init.orthogonal_(param, gain=math.sqrt(2))
            elif 'bias' in name: nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def _encode(self, state):
        device=next(self.parameters()).device; x=state.to(device)
        x=self.input_norm(x); x=F.gelu(self.input_proj(x))
        for b in self.res_blocks: x=b(x)
        for b in self.temporal:   x=b(x)
        return x

    def forward(self, state, legal_actions, hidden_state=None):
        x=self._encode(state); logits=self.actor_head(x)
        mask=torch.full_like(logits,-1e9)
        for a in legal_actions:
            if 0<=a<self.action_size: mask[:,a]=0.0
        return F.softmax(logits+mask,dim=-1), self.critic_head(x), None

    def get_action(self, state, legal_actions, hidden_state=None, deterministic=False):
        probs,value,_=self.forward(state,legal_actions)
        dist=torch.distributions.Categorical(probs)
        action=probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, None

    def evaluate_actions(self, states, legal_actions_batch, actions):
        states_t=torch.stack(list(states)) if isinstance(states,(list,tuple)) else states
        actions_t=torch.stack(list(actions)).view(-1) if isinstance(actions,(list,tuple)) else actions.view(-1)
        N=states_t.shape[0]; device=next(self.parameters()).device
        states_t=states_t.to(device); actions_t=actions_t.to(device)
        mask=torch.full((N,self.action_size),-1e9,device=device)
        for i,legal in enumerate(legal_actions_batch):
            for a in legal:
                if 0<=a<self.action_size: mask[i,a]=0.0
        x=self._encode(states_t)
        probs=F.softmax(self.actor_head(x)+mask,dim=-1)
        dist=torch.distributions.Categorical(probs)
        return dist.log_prob(actions_t), dist.entropy(), self.critic_head(x).squeeze(-1)
