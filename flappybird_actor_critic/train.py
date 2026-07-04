"""
Training script for the FlappyGrid environment using Advantage Actor-Critic (A2C).

Uses Monte Carlo returns with a value baseline (Critic) to compute advantages.
"""

import numpy as np
import torch
import torch.nn.functional as F

from flappy_env import FlappyGrid
from mlp import FlappyGridActorCritic

# --- Hyperparameters ---
LR = 1e-3
GAMMA = 0.99
ENTROPY_COEFF = 0.01
EPISODES = 55_000

env = FlappyGrid()
state_dim = env.W * env.H
n_actions = 3

policy_net = FlappyGridActorCritic(input_dim=state_dim)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

# --- Training Metrics ---
episode_rewards = []
episode_walls = []

best_avg_wall = -float('inf')

def compute_returns(rewards, gamma):
  T = len(rewards)
  returns = np.zeros(T, dtype=np.float32)
  G = 0
  for t in reversed(range(T)):
    G = rewards[t] + gamma * G
    returns[t] = G
  return returns


# --- Training Loop ---
for episode in range(EPISODES):
  state = env.reset()
  done = False
  log_probs = []
  values = []
  all_probs_list = []
  rewards = []
  total_reward = 0
  ep_walls = 0

  while not done:
    state_tensor = torch.FloatTensor(state)
    logits, value = policy_net(state_tensor)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    all_probs_step = probs

    _, reward, done = env.tick(action)
    next_state = env.pos_to_state()

    log_probs.append(log_prob)
    values.append(value)
    all_probs_list.append(all_probs_step)
    rewards.append(reward)
    total_reward += reward
    if reward >= 1:
      ep_walls += 1
    state = next_state

  # Compute returns G_t
  returns_np = compute_returns(rewards, GAMMA)
  returns = torch.tensor(returns_np, dtype=torch.float32)
  log_probs = torch.stack(log_probs)
  values = torch.stack(values).squeeze()
  all_probs = torch.cat(all_probs_list, dim=0)

  # Advantage = G_t - V(s_t)
  advantages = returns - values.detach()

  # Actor loss (mean over timesteps for stability)
  actor_loss = -(log_probs * advantages).mean()
  # Critic loss
  critic_loss = F.mse_loss(values, returns)

  # Entropy bonus to encourage exploration
  entropy = -(all_probs * all_probs.log()).sum(dim=-1).mean()

  # loss = actor_loss + critic_loss
  loss = actor_loss + critic_loss - ENTROPY_COEFF * entropy

  optimizer.zero_grad()
  loss.backward()
  # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
  optimizer.step()

  episode_rewards.append(total_reward)
  episode_walls.append(ep_walls)

  if (episode + 1) % 100 == 0:
    avg_r = np.mean(episode_rewards[-100:])
    avg_w = np.mean(episode_walls[-100:])
    print(
      f"Ep {episode+1}/{EPISODES} | Reward: {avg_r:.1f} | Avg Walls Passed: {avg_w:.1f}")
    
    if avg_w > best_avg_wall:
      torch.save(policy_net.state_dict(), "actor_critic_weights.pt")
      best_avg_wall = avg_w
      print("✅ New best", best_avg_wall)

torch.save(policy_net.state_dict(), "actor_critic_weights.pt")
print("Saved final model to actor_critic_weights.pt")
