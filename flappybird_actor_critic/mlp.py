import torch.nn as nn


class FlappyGridActorCritic(nn.Module):
  """
  A PyTorch-based Multi-Layer Perceptron (MLP) implementing an Actor-Critic architecture.

  This network has a shared trunk (feature extractor) and splits into two heads:
  1. Actor: Outputs raw logits for each possible action (policy).
  2. Critic: Outputs a single value representing the estimated state value V(s).

  Input:  Flattened grid state of shape (batch, H * W)
  Output: 
    - Logits: Raw action logits of shape (batch, n_actions)
    - Value: State value estimate of shape (batch,)
  """

  def __init__(self, input_dim, hidden_dims=[64, 64, 64, 64], n_actions=3):
    super().__init__()
    layers = []
    prev = input_dim
    for h in hidden_dims:
      layers.append(nn.Linear(prev, h))
      layers.append(nn.ReLU())
      prev = h
    self.shared = nn.Sequential(*layers)
    self.actor = nn.Linear(prev, n_actions)   # Logits for action selection
    self.critic = nn.Linear(prev, 1)          # State value estimate V(s)

  def forward(self, x):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    features = self.shared(x)
    logits = self.actor(features)
    value = self.critic(features)
    # Return logits and value (shape [batch])
    return logits, value.squeeze(-1)
