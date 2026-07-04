"""
Evaluation/Play script for the FlappyGrid environment using a trained Actor-Critic model.

This script loads the trained model weights, runs the environment in a loop, and
displays a real-time ASCII rendering of the bird dodging obstacles, alongside
metrics like confidence and actions taken.
"""

import os
import torch
from time import sleep
from flappy_env import FlappyGrid
from mlp import FlappyGridActorCritic

# Action mappings for descriptive output
A = {0: '~', 1: 'flap', 2: 'flap flap'}

env = FlappyGrid()
state_dim = env.W * env.H

# Initialize network and load trained actor-critic weights
net = FlappyGridActorCritic(input_dim=state_dim, n_actions=3)
net.load_state_dict(torch.load("actor_critic_weights.pt", map_location="cpu"))
net.eval()

state = env.reset()
step = 0
walls_passed = 0

# Game loop
while not env.done:
  os.system('clear')

  with torch.no_grad():
    logits, _ = net(torch.FloatTensor(state))
    probs = torch.softmax(logits, dim=-1).squeeze()

  # Action selection: choose the option with the highest confidence
  action = probs.argmax().item()
  confidence = probs.max().item()

  _, reward, done = env.tick(action)
  state = env.pos_to_state()
  if reward >= 1:
    walls_passed += 1

  # Display the grid and log details if the episode is still active
  if not done:
    env.render()
    step += 1

    bar_len = 20
    filled = int(confidence * bar_len)
    bar = '█' * filled + '░' * (bar_len - filled)

    print(f"  Step: {step}  |  Walls: {walls_passed}  |  Action: {A[action]}")
    # print(f"  Confidence:   [{bar}] {confidence:.2f}")

  sleep(0.3)

print(f"\nGame over — survived {step} steps, passed {walls_passed} walls")
