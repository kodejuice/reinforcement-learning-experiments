import numpy as np
import torch
import torch.nn.functional as F

from paddle_env import PaddleCatch
from mlp import ContinuousActorCritic

# --- Hyperparameters ---
LR = 1e-3
GAMMA = 0.99
ENTROPY_COEFF = 0.01
EPISODES = 23_000

env = PaddleCatch()
policy_net = ContinuousActorCritic()
# policy_net.load_state_dict(torch.load("paddle_catch_weights.pt", map_location="cpu"))

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
best_avg_r = -float('inf')


def compute_returns(rewards, gamma):
  """
  Computes the discounted returns for a sequence of rewards.

  Args:
      rewards (list): A list of rewards received in an episode.
      gamma (float): The discount factor [0, 1].

  Returns:
      np.ndarray: An array of discounted returns for each time step.
  """
  T = len(rewards)
  returns = np.zeros(T, dtype=np.float32)
  G = 0
  for t in reversed(range(T)):
    G = rewards[t] + gamma * G
    returns[t] = G
  return returns

episode_rewards = []

# --- Training Loop ---
for episode in range(EPISODES):
  state = env.reset()
  done = False
  log_probs = []
  values = []
  rewards = []
  entropies = []

  while not done:
    state_tensor = torch.FloatTensor(state)

    mu, std, value = policy_net(state_tensor)
    dist = torch.distributions.Normal(mu, std)
    action: torch.Tensor = dist.sample()
    action_clipped = torch.clamp(action, -1, 1)
    # action_clipped = action.clamp(-1, 1)

    log_prob = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy()

    next_state, reward, done = env.step(action_clipped.item())

    log_probs.append(log_prob)
    values.append(value)
    rewards.append(reward)
    entropies.append(entropy)

    state = next_state

  # Compute returns G_t
  returns_np = compute_returns(rewards, GAMMA)
  returns = torch.tensor(returns_np, dtype=torch.float32)
  log_probs = torch.stack(log_probs)
  values = torch.stack(values).squeeze()
  entropy = torch.stack(entropies).mean()

  # Advantage = G_t - V(s_t)
  advantages = returns - values.detach()

  # Actor loss (mean over timesteps for stability)
  actor_loss = -(log_probs * advantages).mean()
  # Critic loss
  critic_loss = F.mse_loss(values, returns)

  loss = actor_loss + critic_loss - ENTROPY_COEFF * entropy

  optimizer.zero_grad()
  loss.backward()
  # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
  optimizer.step()

  # return from first step is total discounted reward
  total_reward = returns_np[0]
  episode_rewards.append(total_reward)

  if (episode + 1) % 100 == 0:
    avg_r = np.mean(episode_rewards[-100:])
    print(f"Ep {episode+1}/{EPISODES} | Reward: {avg_r:.1f}")

    if avg_r > best_avg_r:
      torch.save(policy_net.state_dict(), "paddle_catch_weights.pt")
      best_avg_r = avg_r
      print("✅ New best", best_avg_r)

torch.save(policy_net.state_dict(), "paddle_catch_weights.pt")
print("Saved final model to paddle_catch_weights.pt")
