# Paddle Catch (Continuous Actor-Critic)

![Paddle Animation](paddle%20animation.gif)

A PyTorch-based Advantage Actor-Critic (A2C) agent that learns to play a simple continuous-action Paddle Catch game.

## Overview

The environment `PaddleCatch` is a continuous environment where the agent controls a paddle to catch a bouncing ball.

- **State**: A 5-dimensional vector representing:
  - Ball X, Y position (0 to 1)
  - Ball X, Y velocities
  - Paddle X center position (0 to 1)
- **Actions**: A continuous scalar in $[-1, 1]$ representing the paddle velocity. Positive moves right, negative moves left.
- **Rewards**:
  - +10.0 for successfully catching the ball.
  - -10.0 for missing the ball (ball hits bottom wall without paddle overlap).
  - -0.1 small step penalty to encourage efficiency.

## Files

- [`paddle_env.py`](./paddle_env.py) - The continuous environment logic and text rendering.
- [`mlp.py`](./mlp.py) - A multi-layer perceptron implementing the Actor-Critic network with a learnable, state-independent standard deviation.
- [`train.py`](./train.py) - The training loop. Implements A2C for continuous action spaces using a Normal distribution.
- [`play.py`](./play.py) - An inference script that loads trained weights (`paddle_catch_weights.pt`) and renders the agent playing the game.

## Usage

**To train the agent:**

```bash
python3 train.py
```

**To watch the agent play:**

```bash
python3 play.py
```

This requires a trained `paddle_catch_weights.pt` file in the same directory.

## Training Loop (Advantage Actor-Critic for Continuous Actions)

The agent is trained using the Advantage Actor-Critic (A2C) algorithm. In each episode:

1. **Collect Trajectories:** The agent plays an episode, collecting states $s_t$, continuous actions $a_t$, and rewards $r_t$.
2. **Compute Returns:** For each time step $t$, we compute the discounted return $G_t$:
   $$ G*t = \sum*{k=0}^{\infty} \gamma^k r\_{t+k} $$
3. **Calculate Advantage:** The advantage $A_t$ is computed using the Critic network's value estimate $V(s_t)$:
   $$ A_t = G_t - V(s_t) $$
4. **Compute Losses:**
   - **Actor Loss**: Encourages actions that led to better-than-expected returns. The action probabilities are modeled as a Normal distribution $\mathcal{N}(\mu, \sigma^2)$ where $\mu$ is output by the network and $\sigma$ is a learned parameter.
     $$ \mathcal{L}_{actor} = -\frac{1}{T} \sum_{t=1}^{T} \log \pi(a_t | s_t) A_t $$
   - **Critic Loss**: Minimizes the mean squared error between the value estimate and actual returns.
     $$ \mathcal{L}_{critic} = \frac{1}{T} \sum_{t=1}^{T} (V(s_t) - G_t)^2 $$
   - **Entropy Bonus**: Encourages exploration by penalizing deterministic policies.
     $$ \mathcal{H} = -\frac{1}{T} \sum\_{t=1}^{T} \text{Entropy}(\mathcal{N}(\mu_t, \sigma^2)) $$
5. **Update Networks:** The total loss is backpropagated to update both networks simultaneously:
   $$ \mathcal{L}_{total} = \mathcal{L}_{actor} + \mathcal{L}\_{critic} - c \cdot \mathcal{H} $$
   where $c$ is the entropy coefficient.
