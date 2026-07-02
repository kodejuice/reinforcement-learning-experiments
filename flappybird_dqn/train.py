from opt_einsum.backends import torch
import random
import numpy as np

import torch
import torch.nn as nn

from flappy_env import FlappyGrid
from mlp import FlappyGridNetwork
from replay_buffer import ReplayBuffer

# --- Hyperparameters ---
LR = 5e-4
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_CAPACITY = 20_000
MIN_BUFFER_SIZE = 500
TARGET_UPDATE_FREQ = 1000
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999
EPISODES = 55_000

env = FlappyGrid()
state_dim = env.W * env.H
n_actions = 3

# Two copies of the network: online and target
online_net = FlappyGridNetwork(input_dim=state_dim, output_dim=n_actions)
try:
  online_net.load_state_dict(torch.load("dqn_weights.pt"))
except:
  pass

target_net = FlappyGridNetwork(input_dim=state_dim, output_dim=n_actions)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY, state_dim)

# --- Training loop ---
global_step = 0
episode_rewards = []
episode_walls = []

for episode in range(EPISODES):
  state = env.reset()
  total_reward = 0
  ep_walls = 0
  done = False

  while not done:
    # epsilon-greedy
    if random.random() < EPSILON:
      action = random.randint(0, n_actions - 1)
    else:
      with torch.no_grad():
        q = online_net(torch.FloatTensor(state))
        action = q.argmax().item()

    _, reward, done = env.tick(action)
    next_state = env.pos_to_state()
    total_reward += reward
    if reward >= 1:
      ep_walls += 1

    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state

    # train
    if len(replay_buffer) >= MIN_BUFFER_SIZE:
      b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(BATCH_SIZE)

      states = torch.FloatTensor(b_s)
      actions = torch.LongTensor(b_a)
      rewards = torch.FloatTensor(b_r)
      next_states = torch.FloatTensor(b_ns)
      dones = torch.FloatTensor(b_d)

      # Q-values for chosen actions
      q_values = online_net(states)
      q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()

      # target: r + gamma * max Q(s', a')
      with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1).values
        target = rewards + GAMMA * max_next_q * (1.0 - dones)

      loss = nn.MSELoss()(q_action, target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      global_step += 1
      if global_step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

  episode_rewards.append(total_reward)
  episode_walls.append(ep_walls)
  EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

  if (episode + 1) % 100 == 0:
    avg_r = np.mean(episode_rewards[-100:])
    avg_w = np.mean(episode_walls[-100:])
    print(
      f"Ep {episode+1}/{EPISODES} | Reward: {avg_r:.1f} | Avg Walls Passed: {avg_w:.1f} | ε: {EPSILON:.3f}")
    torch.save(online_net.state_dict(), "dqn_weights.pt")

torch.save(online_net.state_dict(), "dqn_weights.pt")
print("Saved to dqn_weights.pt")
