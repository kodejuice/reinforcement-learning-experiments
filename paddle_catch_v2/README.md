# Paddle Catch V2 (Proximal Policy Optimization)

A PyTorch-based Proximal Policy Optimization (PPO) agent that learns to juggle multiple balls in a complex continuous-action Paddle Catch environment with gravity, dynamic wind, and obstacles.

## Overview

The environment `PaddleCatchV2` is a continuous control environment where the agent controls a paddle to keep multiple balls in the air. The physics engine includes gravity, a random-walk wind force, and static rectangular obstacles that the balls bounce off of.

- **State**: A 10-dimensional vector representing:
  - Paddle X center position (0 to 1)
  - Current Wind X velocity
  - Ball 1: X, Y position, X, Y velocities
  - Ball 2: X, Y position, X, Y velocities
- **Actions**: A continuous scalar in $[-1, 1]$ representing the paddle velocity. Positive moves right, negative moves left.
- **Rewards**:
  - +10.0 for successfully catching a ball and bouncing it back up.
  - -10.0 for missing a ball (ball hits bottom wall without paddle overlap).

## Files

- [`paddle_env.py`](./paddle_env.py) - The environment logic and console-based rendering. Handles multi-ball physics, wind, gravity, and obstacle collisions.
- [`mlp.py`](./mlp.py) - A multi-layer perceptron implementing the Actor-Critic network with a 10-dimensional input and a learnable, state-independent standard deviation.
- [`train.py`](./train.py) - The training loop. Implements Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) for continuous action spaces.
- [`play.py`](./play.py) - An inference script that loads trained weights (`ppo_weights.pt`) and renders the agent playing the game.

## Usage

**To train the agent:**

```bash
python3 train.py
```

**To watch the agent play:**

```bash
python3 play.py
```

This requires a trained `ppo_weights.pt` file in the same directory.

## Training Loop (Proximal Policy Optimization)

The agent is trained using the Proximal Policy Optimization (PPO) algorithm, which improves upon standard Actor-Critic by allowing safe reuse of trajectory data through a clipped surrogate objective.

1. **Collect Trajectories:** The agent plays the environment for a fixed batch of steps (e.g., 2048), collecting states $s_t$, continuous actions $a_t$, rewards $r_t$, and the old log-probabilities of those actions $\log \pi_{\theta_{\text{old}}}(a_t | s_t)$.
2. **Compute Advantages (GAE):** Generalized Advantage Estimation (GAE) is used to compute smoothly blended advantage estimates $A_t$ that balance variance and bias.
3. **PPO Update Epochs:** For $K$ epochs, we shuffle the collected batch and process it in minibatches to update the network:
   - **Ratio Calculation:** We calculate how much the policy has changed for a specific action:
   ```math
   r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})
   ```

   - **Clipped Actor Loss**: The objective is clipped to prevent the new policy from deviating too far from the old policy, ensuring monotonic improvement and preventing destructive updates:
   ```math
   \mathcal{L}_{actor} = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
   ```

   - **Critic Loss**: Minimizes the mean squared error between the value estimate and actual returns.
   - **Entropy Bonus**: Encourages exploration by penalizing deterministic policies.
4. **Update Networks:** The total loss is backpropagated to update both the actor and critic networks simultaneously:

```math
\mathcal{L}_{total} = \mathcal{L}_{actor} + \frac{1}{2}\mathcal{L}_{critic} - c \cdot \mathcal{H}
```

where $c$ is the entropy coefficient.
