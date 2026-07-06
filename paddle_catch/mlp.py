import torch
import torch.nn as nn


class ContinuousActorCritic(nn.Module):
  def __init__(self, state_dim=5, action_dim=1, hidden=64):
    super().__init__()
    self.shared = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU()
    )
    self.actor_mu = nn.Linear(hidden, action_dim)
    self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # learnable, independent of state
    self.critic = nn.Linear(hidden, 1)

  def forward(self, x):
    features = self.shared(x)
    mu = self.actor_mu(features)
    std = torch.exp(self.actor_log_std)
    value = self.critic(features).squeeze(-1)
    return mu, std, value
