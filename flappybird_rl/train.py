import numpy as np
from flappy_env import FlappyGrid

# --- Q-Learning Hyperparameters ---
ALPHA = 0.1        # learning rate
GAMMA = 0.9        # discount factor
EPSILON = 1.0      # initial exploration rate
EPSILON_MIN = 0.01  # minimum exploration
EPSILON_DECAY = 0.999  # decay per episode
EPISODES = 200_000

env = FlappyGrid(dim=(15, 25))
n_actions = 3

# Dimensions: bird_row, distance_to_wall, closest_wall_end, second_wall_end, gravity_phase, action
Q = np.zeros((env.H, env.W, env.H, env.H, 2, n_actions))

print(f"Starting Q-learning training for {EPISODES} episodes...")
print(f"Q-Table Shape: {Q.shape} ({Q.size} parameters)")

episode_rewards = []
episode_walls = []

for episode in range(EPISODES):
  state = env.reset()
  total_reward = 0
  ep_walls = 0
  done = False

  while not done:
    # ε-greedy action selection
    if np.random.random() < EPSILON:
      action = np.random.randint(n_actions)
    else:
      action = np.argmax(Q[state])

    next_state, reward, done = env.tick(action)
    total_reward += reward
    if reward == 1:
      ep_walls += 1

    s_a = tuple(list(state) + [action])

    # Q-learning Bellman update
    best_next = np.max(Q[next_state]) if not done else 0
    target = reward + GAMMA * best_next
    Q[s_a] += ALPHA * (target - Q[s_a])

    state = next_state

  episode_rewards.append(total_reward)
  episode_walls.append(ep_walls)
  EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

  if (episode + 1) % 1000 == 0:
    avg_reward = np.mean(episode_rewards[-100:])
    avg_walls = np.mean(episode_walls[-100:])
    print(f"Episode {episode + 1}/{EPISODES} | Avg Reward: {avg_reward:.1f} | Avg Walls Survived: {avg_walls:.1f} | Epsilon: {EPSILON:.3f}")

# Save the trained policy
np.save("q_table.npy", Q)
print("Training completed. Learned policy saved to 'q_table.npy'.")
