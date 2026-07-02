import torch.nn as nn


class FlappyGridNetwork(nn.Module):
  """
  A PyTorch-based Multi-Layer Perceptron (MLP) for Q-value approximation.

  Takes the flattened grid state as input and outputs the raw Q-values for
  each possible action.

  Input:  flattened grid (H * W)
  Output: 3 Q-values (NOTHING, FLAP, FLAP_FLAP)
  """

  def __init__(self, input_dim, hidden_dims=[32, 32, 32, 32], output_dim=3):
    super().__init__()
    layers = []
    prev = input_dim
    for h in hidden_dims:
      layers.append(nn.Linear(prev, h))
      layers.append(nn.ReLU())
      prev = h
    layers.append(nn.Linear(prev, output_dim))  # no activation, raw Q-values
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    return self.net(x)
