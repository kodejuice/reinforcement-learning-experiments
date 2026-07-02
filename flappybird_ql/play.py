import os
import sys
from time import sleep
import numpy as np
from flappy_env import FlappyGrid, A


def clear_console():
  os.system('cls' if os.name == 'nt' else 'clear')


def main():
  if not os.path.exists("q_table.npy"):
    print("Error: 'q_table.npy' not found.")
    print("Please train the model first by running: python3 train.py")
    sys.exit(1)

  # Load the trained Q-table
  Q = np.load("q_table.npy")
  env = FlappyGrid(dim=(15, 25))

  env.reset()
  step = 0
  walls_passed = 0
  total_reward = 0

  print("Running inference. Press Ctrl+C to stop.")
  sleep(1)

  while not env.done:
    state = env.pos_to_state()
    q_vals = Q[state]
    best_action = np.argmax(q_vals)
    confidence = q_vals[best_action] - np.mean(q_vals)

    clear_console()
    env.render()

    bar_len = int(min(max(confidence, 0), 5) * 4)  # scale to max 20 chars
    bar = '█' * bar_len + '░' * (20 - bar_len)

    print(
      f" Step: {step}  |  Walls passed: {walls_passed}  |  Action: {A[best_action]}")
    print(f" Q-confidence: [{bar}] {confidence:.2f}")

    _, reward, _ = env.tick(best_action)
    if reward == 1:
      walls_passed += 1
    total_reward += reward
    step += 1
    sleep(0.3)

  print(
    f"\nGame Over! Total steps: {step}, Walls passed: {walls_passed}")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nExiting inference play session.")
