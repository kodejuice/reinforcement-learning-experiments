import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from paddle_env import PaddleCatchV2
from mlp import ContinuousActorCritic

# --- Hyperparameters ---
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
ENTROPY_COEFF = 0.01
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64
K_EPOCHS = 10
TOTAL_TIMESTEPS = 4_000_000

env = PaddleCatchV2()
policy_net = ContinuousActorCritic(state_dim=10, action_dim=1, hidden=64)
# policy_net.load_state_dict(torch.load("ppo_weights.pt", map_location="cpu"))

optimizer = optim.Adam(policy_net.parameters(), lr=LR)


def compute_gae(rewards, values, dones, gamma, lam, next_value):
  """
  Compute Generalized Advantage Estimation (GAE).
  """
  advantages = np.zeros_like(rewards, dtype=np.float32)
  lastgaelam = 0
  for t in reversed(range(len(rewards))):
    if t == len(rewards) - 1:
      nextnonterminal = 1.0 - dones[t]
      nextvalues = next_value
    else:
      nextnonterminal = 1.0 - dones[t]
      nextvalues = values[t + 1]

    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
    advantages[t] = lastgaelam = delta + \
        gamma * lam * nextnonterminal * lastgaelam

  returns = advantages + values
  return advantages, returns


state = env.reset()
global_step = 0
best_avg_r = -12.3
# best_avg_r = -float('inf')

episode_rewards = []
current_ep_reward = 0

print("Starting PPO Training...")

while global_step < TOTAL_TIMESTEPS:
  b_states = []
  b_actions = []
  b_logprobs = []
  b_rewards = []
  b_dones = []
  b_values = []

  # Collection phase
  for _ in range(BATCH_SIZE):
    global_step += 1

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
      mu, std, value = policy_net(state_tensor)
      dist = torch.distributions.Normal(mu, std)
      action = dist.sample()
      log_prob = dist.log_prob(action).sum(dim=-1)

    action_clipped = torch.clamp(action, -1.0, 1.0).item()
    next_state, reward, done = env.step(action_clipped)

    b_states.append(state)
    b_actions.append(action.item())
    b_logprobs.append(log_prob.item())
    b_rewards.append(reward)
    b_dones.append(done)
    b_values.append(value.item())

    current_ep_reward += reward
    state = next_state

    if done:
      episode_rewards.append(current_ep_reward)
      current_ep_reward = 0
      state = env.reset()

  # Calculate GAE
  with torch.no_grad():
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    _, _, next_value = policy_net(state_tensor)
    next_value = next_value.item()

  b_advantages, b_returns = compute_gae(
      np.array(b_rewards),
      np.array(b_values),
      np.array(b_dones),
      GAMMA,
      GAE_LAMBDA,
      next_value
  )

  # Convert to tensors
  b_states = torch.FloatTensor(np.array(b_states))
  b_actions = torch.FloatTensor(np.array(b_actions)).unsqueeze(1)
  b_logprobs = torch.FloatTensor(np.array(b_logprobs))
  b_advantages = torch.FloatTensor(b_advantages)
  b_returns = torch.FloatTensor(b_returns)

  # Normalize advantages
  b_advantages = (b_advantages - b_advantages.mean()) / \
      (b_advantages.std() + 1e-8)

  # Update phase
  dataset = TensorDataset(
    b_states, b_actions, b_logprobs, b_advantages, b_returns)
  dataloader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True)

  for epoch in range(K_EPOCHS):
    for batch in dataloader:
      mb_states, mb_actions, mb_old_logprobs, mb_advantages, mb_returns = batch

      mu, std, value = policy_net(mb_states)
      dist = torch.distributions.Normal(mu, std)
      mb_new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
      entropy = dist.entropy().sum(dim=-1).mean()

      # Policy loss
      ratio = torch.exp(mb_new_logprobs - mb_old_logprobs)
      surr1 = ratio * mb_advantages
      surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * mb_advantages
      actor_loss = -torch.min(surr1, surr2).mean()

      # Value loss
      critic_loss = F.mse_loss(value, mb_returns)

      # Total loss
      loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * entropy

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
      optimizer.step()

  # Logging
  if len(episode_rewards) > 0:
    avg_r = np.mean(episode_rewards[-100:])
    print(f"Step {global_step:8d} | Avg Reward (last 100): {avg_r:.1f}")

    if avg_r > best_avg_r:
      torch.save(policy_net.state_dict(), "ppo_weights.pt")
      best_avg_r = avg_r
      print(f"✅ New best: {best_avg_r:.1f}")

# torch.save(policy_net.state_dict(), "ppo_weights.pt")
print("Training finished. Model saved.")
