# FlappyGrid RL — Tabular Q-Learning Flappy Bird

![Flappy Bird Play](flappy%20play.gif)

A lightweight, terminal-based grid world Flappy Bird environment solved using Tabular Q-Learning. 

The environment features dynamic physical constraints (gravity phases, ceiling collision damping) and dual moving obstacles that require foresight to dodge.

---

## Mechanics & Physics

- **Agent**: The bird sits statically in column `0` and adjusts its vertical position.
- **Walls**: Two vertical walls (`█`), each occupying 50% of the grid height. They spawn offset by half the grid width and:
  - Move left `1` column per tick.
  - Drift downward by `1` row per tick.
  - Wrap around vertically (ensuring a continuous 30% gap exists somewhere on the grid).
- **Gravity**: Pulls the bird down by `1` row every other tick (odd ticks) if no action is taken.
- **Actions**:
  - `0` (NOTHING) — Let gravity act.
  - `1` (FLAP) — Ascend `1` row.
  - `2` (FLAP_FLAP) — Ascend `2` rows.
- **Boundaries**:
  - **Floor**: Reaching row `H-1` results in terminal death.
  - **Ceiling**: Bumping into row `0` keeps the bird at row `0` but inflicts a gentle reward penalty (`-1`) to discourage ceiling-hugging.

---

## State Representation

A naive representation of 2 walls on screen (bird row, wall 1 col, wall 1 gap, wall 2 col, wall 2 gap) would create a massive state space ($H \times W \times H \times W \times H \times 2$ states), making tabular Q-learning extremely slow.

To optimize, **FlappyGrid sorts the walls by proximity** and only exposes the closest wall's distance. Because the two walls maintain a constant horizontal spacing of half the grid width, the horizontal distance of the second wall is fully deterministic. 

The state is represented as a 5-tuple:
$$\text{State} = (y_{\text{bird}}, x_{\text{wall}}, y_{\text{gap1}}, y_{\text{gap2}}, \phi_{\text{grav}})$$

1. **`bird_pos`** ($y_{\text{bird}}$): Bird's current row index `[0..H-1]`.
2. **`distance_to_wall`** ($x_{\text{wall}}$): Column index of the nearest wall `[0..W-1]`.
3. **`wall_end_row`** ($y_{\text{gap1}}$): Bottom row index of the nearest wall's gap `[0..H-1]`.
4. **`second_wall_end_row`** ($y_{\text{gap2}}$): Bottom row index of the second nearest wall's gap `[0..H-1]`.
5. **`gravity_phase`** ($\phi_{\text{grav}}$): A binary phase bit (`0` or `1`) indicating whether gravity will pull the bird down on the next tick.

This reduces the total state space size to only $H \times W \times H \times H \times 2$ states (a Q-table size of $H \times W \times H \times H \times 2 \times 3$ parameters), allowing complete convergence.

---

## Reward Structure

- **Wall Passed**: `+1` (awarded the tick after a wall column reaches `0` without collision).
- **Collision (Wall or Floor)**: `-10` (terminal event).
- **Ceiling Bump**: `-1` (non-terminal penalty).

---

## Hyperparameters

- **Learning Rate ($\alpha$)**: `0.1`
- **Discount Factor ($\gamma$)**: `0.9`
- **Episodes**: `200,000`
- **Exploration ($\epsilon$)**: Starts at `1.0` and decays by `0.999` per episode down to a minimum of `0.01`.

## Training Loop (Tabular Q-Learning)

The agent is trained using standard tabular Q-Learning. In each step of an episode:

1. **Observe State:** The agent observes the current state $s_t$.
2. **Choose Action:** An action $a_t$ is selected using an $\epsilon$-greedy policy based on the current Q-table $Q(s_t, a)$.
3. **Execute Action:** The action is taken in the environment, yielding reward $r_t$ and the next state $s_{t+1}$.
4. **Update Q-value:** The Q-table entry for $(s_t, a_t)$ is updated using the Bellman equation:
   $$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$
   where $\alpha$ is the learning rate and $\gamma$ is the discount factor.
5. **Decay Epsilon:** After each episode, the exploration rate $\epsilon$ is decayed to gradually shift from exploration to exploitation.

---

## How to Run

### 1. Training the Model
To train the Q-learning agent, run the training script:
```bash
python3 train.py
```
This runs the episodes in the terminal and outputs progress periodically:
```text
Episode 1000/100000 | Avg Reward: -9.8 | Avg Walls Survived: 0.9 | Epsilon: 0.368
Episode 2000/100000 | Avg Reward: -8.1 | Avg Walls Survived: 2.1 | Epsilon: 0.135
...
```
Once training finishes, the learned Q-table is saved to `q_table.npy`.

### 2. Playing Inference
To run a live simulation of the trained agent navigating through the obstacles:
```bash
python3 play.py
```
This clears the terminal screen and renders the game frame-by-frame.
