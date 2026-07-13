import torch
import torch.nn as nn


class ContinuousActorCritic(nn.Module):
  def __init__(self, state_dim=10, action_dim=1, hidden=64):
    super().__init__()
    # Shared features
    self.shared = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh()
    )

    # Actor
    self.actor_mu = nn.Linear(hidden, action_dim)
    # Learnable standard deviation, independent of state
    self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

    # Critic
    self.critic = nn.Linear(hidden, 1)

    # Initialize weights for stability
    nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)
    nn.init.constant_(self.actor_mu.bias, 0)
    nn.init.orthogonal_(self.critic.weight, gain=1.0)
    nn.init.constant_(self.critic.bias, 0)

  def forward(self, x):
    features = self.shared(x)

    # Actor outputs
    mu = self.actor_mu(features)
    # Expand log_std to match batch size if necessary
    std = torch.exp(self.actor_log_std).expand_as(mu)

    # Critic output
    value = self.critic(features).squeeze(-1)

    return mu, std, value
