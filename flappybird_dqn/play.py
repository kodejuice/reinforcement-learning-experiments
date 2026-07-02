import os
import torch
from time import sleep
from flappy_env import FlappyGrid
from mlp import FlappyGridNetwork

A = {0: '~', 1: 'flap', 2: 'flap flap'}

env = FlappyGrid()
state_dim = env.W * env.H

net = FlappyGridNetwork(input_dim=state_dim, output_dim=3)
net.load_state_dict(torch.load("dqn_weights.pt"))
net.eval()

state = env.reset()
step = 0
walls_passed = 0

while not env.done:
  os.system('clear')

  q = net(torch.FloatTensor(state)).squeeze()
  action = q.argmax().item()
  confidence = (q.max() - q.mean()).item()

  _, reward, done = env.tick(action)
  state = env.pos_to_state()
  if reward >= 1:
    walls_passed += 1

  if not done:
    env.render()
    step += 1

    bar_len = 20
    filled = int(min(confidence, 5) / 5 * bar_len)
    bar = '█' * filled + '░' * (bar_len - filled)

    print(f"  Step: {step}  |  Walls: {walls_passed}  |  Action: {A[action]}")
    print(f"  Q-confidence: [{bar}] {confidence:.2f}")

  sleep(0.3)

print(f"\nGame over — survived {step} steps, passed {walls_passed} walls")
