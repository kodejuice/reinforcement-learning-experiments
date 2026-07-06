import time
import torch
import os
from paddle_env import PaddleCatch
from mlp import ContinuousActorCritic


def play(episodes=20):
  """
  Evaluates the trained Actor-Critic agent by having it play the game visually.
  Loads the learned weights from 'paddle_catch_weights.pt'.
  During evaluation, the agent acts deterministically by using the mean action.

  Args:
      episodes (int): The number of episodes to play.
  """
  env = PaddleCatch()
  policy_net = ContinuousActorCritic()

  try:
    policy_net.load_state_dict(torch.load("paddle_catch_weights.pt"))
    print("Loaded trained weights from 'paddle_catch_weights.pt'.")
  except FileNotFoundError:
    print("No weights found! Playing with random untrained weights.")

  # Set to evaluation mode
  policy_net.eval()

  for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # Clear screen for animation (works on mac/linux)
    os.system('clear')
    print(f"--- Episode {ep+1} ---")
    env.render()
    time.sleep(0.5)

    while not done:
      state_tensor = torch.FloatTensor(state)

      with torch.no_grad():
        # For inference, we use the deterministic mean action (mu)
        # and ignore the standard deviation (std)
        mu, _, _ = policy_net(state_tensor)

      # Clamp action to [-1, 1] just in case, though env also clips it
      action = torch.clamp(mu, -1.0, 1.0).item()

      state, reward, done = env.step(action)
      total_reward += reward

      os.system('clear')
      print(f"--- Episode {ep+1} ---")
      env.render()
      time.sleep(0.05)  # Pause briefly to see the animation

    print(f"\nEpisode {ep+1} finished! Total Reward: {total_reward:.2f}")
    time.sleep(1.5)  # Pause before starting the next episode


if __name__ == "__main__":
  play()
