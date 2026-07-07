# Flappy Bird (Actor-Critic)

![Flappy Bird Play](flappy%20play.gif)

A PyTorch-based Advantage Actor-Critic (A2C) agent that learns to play a grid-world version of Flappy Bird.

## Overview

The environment `FlappyGrid` is a 15x30 grid. The bird (🦉) must navigate through gaps in moving walls (██) while dealing with gravity that pulls it down every tick.

- **State**: A flattened array representing the visual grid (1s for walls, 0.5 for the bird, 0 for empty space).
- **Actions**: 3 discrete actions: `NOTHING` (0), `FLAP` (1), `FLAP_FLAP` (2).
- **Rewards**:
  - +1 for successfully passing a wall.
  - -3 for hitting a wall or the floor.
  - -1 for hitting the ceiling.

## Files

- [`flappy_env.py`](./flappy_env.py) - The environment logic and rendering.
- [`mlp.py`](./mlp.py) - A simple multi-layer perceptron (MLP) implementing the Actor-Critic network.
- [`train.py`](./train.py) - The training loop. Implements the Advantage Actor-Critic algorithm with state-value baseline estimates.
- [`play.py`](./play.py) - An inference script that loads trained weights (`actor_critic_weights.pt`) and renders the agent playing the game in the terminal.

## Usage

**To train the agent:**

```bash
python3 train.py
```

**To watch the agent play:**

```bash
python3 play.py
```

This requires a trained `actor_critic_weights.pt` file in the same directory.

## Training Loop (Advantage Actor-Critic)

The agent is trained using the Advantage Actor-Critic (A2C) algorithm. In each episode:

1. **Collect Trajectories:** The agent plays an episode, collecting states $s_t$, actions $a_t$, and rewards $r_t$.
2. **Compute Returns:** For each time step $t$, we compute the discounted return $G_t$:
```math
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
```
3. **Calculate Advantage:** The advantage $A_t$ is computed using the Critic network's value estimate $V(s_t)$:
```math
A_t = G_t - V(s_t)
```
4. **Compute Losses:**
   - **Actor Loss**: Encourages actions that led to better-than-expected returns.
```math
\mathcal{L}_{actor} = -\frac{1}{T} \sum_{t=1}^{T} \log \pi(a_t | s_t) A_t
```
   - **Critic Loss**: Minimizes the mean squared error between the value estimate and actual returns.
```math
\mathcal{L}_{critic} = \frac{1}{T} \sum_{t=1}^{T} (V(s_t) - G_t)^2
```
   - **Entropy Bonus**: Encourages exploration by penalizing deterministic policies.
```math
\mathcal{H} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{a} \pi(a | s_t) \log \pi(a | s_t)
```
5. **Update Networks:** The total loss is backpropagated to update both networks simultaneously:
```math
\mathcal{L}_{total} = \mathcal{L}_{actor} + \mathcal{L}_{critic} - c \cdot \mathcal{H}
```
   where $c$ is the entropy coefficient.
