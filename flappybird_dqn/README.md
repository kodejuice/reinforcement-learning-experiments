# Flappy Bird (DQN)

A PyTorch-based Deep Q-Network (DQN) agent that learns to play a grid-world version of Flappy Bird.

## Overview

The environment `FlappyGrid` is a 15x30 grid. The bird (🦉) must navigate through gaps in moving walls (██) while dealing with gravity that pulls it down every tick.

- **State**: A flattened array representing the visual grid (1s for walls, 0.5 for the bird, 0 for empty space).
- **Actions**: 3 discrete actions: `NOTHING` (0), `FLAP` (1), `FLAP_FLAP` (2).
- **Rewards**:
  - +5 for successfully passing a wall.
  - -15 for hitting a wall or the floor.
  - -1 for hitting the ceiling.

## Files

- `flappy_env.py` - The environment logic and rendering.
- `mlp.py` - A simple multi-layer perceptron (MLP) built with PyTorch `nn.Module`.
- `replay_buffer.py` - A fast, numpy-based circular replay buffer for experience replay.
- `train.py` - The training loop. Implements epsilon-greedy exploration, Adam optimization, and target network updates.
- `play.py` - An inference script that loads trained weights (`dqn_weights.pt`) and renders the agent playing the game in the terminal.

## Usage

**To train the agent:**

```bash
python3 train.py
```

This will train for 55,000 episodes (or until interrupted) and save the weights to `dqn_weights.pt`.

**To watch the agent play:**

```bash
python3 play.py
```

This requires a trained `dqn_weights.pt` file in the same directory.
