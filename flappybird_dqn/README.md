# Flappy Bird (DQN)

![Flappy Bird Play](flappy%20play.gif)

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

- [`flappy_env.py`](./flappy_env.py) - The environment logic and rendering.
- [`mlp.py`](./mlp.py) - A simple multi-layer perceptron (MLP) built with PyTorch `nn.Module`.
- [`replay_buffer.py`](./replay_buffer.py) - A fast, numpy-based circular replay buffer for experience replay.
- [`train.py`](./train.py) - The training loop. Implements epsilon-greedy exploration, Adam optimization, and target network updates.
- [`play.py`](./play.py) - An inference script that loads trained weights (`dqn_weights.pt`) and renders the agent playing the game in the terminal.

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

## Training Loop (Deep Q-Network)

The agent is trained using the Deep Q-Network (DQN) algorithm with experience replay and a target network.

1. **Collect Experience:** The agent interacts with the environment using an $\epsilon$-greedy policy. Transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a replay buffer.
2. **Sample Mini-batch:** A random mini-batch of transitions is sampled from the replay buffer to break correlation between consecutive samples.
3. **Compute Target Q-values:** For each transition, the target Q-value $Y_t$ is computed using the target network (which provides stable targets):
   $$ Y_t = r_t + \gamma \max_{a'} Q_{target}(s_{t+1}, a') \cdot (1 - d_t) $$
   where $d_t$ is the boolean done flag (1 if episode ended, 0 otherwise).
4. **Compute Loss:** The Mean Squared Error (MSE) loss is computed between the predicted Q-values from the online network and the target Q-values:
   $$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( Q_{online}(s_i, a_i) - Y_i \right)^2 $$
5. **Update Network:** The loss is backpropagated to update the online network $Q_{online}$.
6. **Target Network Sync:** Every $C$ steps, the target network's weights are completely overwritten with the online network's weights.
