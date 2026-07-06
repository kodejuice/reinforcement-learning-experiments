import numpy as np


class PaddleCatch:
  """
  A simple continuous-action environment: catch a bouncing ball with a paddle.

  State (5-dim vector): [ball_x, ball_y, ball_vx, ball_vy, paddle_x]
      - ball_x, ball_y ∈ [0, 1]
      - ball velocities can be negative or positive
      - paddle_x ∈ [0, 1] (center of paddle)

  Action: scalar in [-1, 1] representing paddle velocity.
      - Positive moves paddle right, negative moves left.

  Termination:
      - Ball hits paddle (y ≤ 0 + margin, and paddle overlaps ball_x) → +10
      - Ball hits bottom wall (y ≤ 0, no paddle overlap) → -10
      - Small step penalty (-0.1) to encourage efficiency.

  Walls: top, left, right reflect the ball. Paddle bounces off left/right boundaries.
  """

  def __init__(self, paddle_width=0.2, paddle_speed=0.03, dt=0.05):
    """
    Initializes the PaddleCatch environment.

    Args:
        paddle_width (float): Width of the paddle relative to the environment width (0 to 1).
        paddle_speed (float): Maximum distance the paddle can move per step.
        dt (float): Time step size for the physics simulation.
    """
    self.paddle_width = paddle_width
    self.paddle_speed = paddle_speed  # max paddle velocity per step
    self.dt = dt
    self.reset()

  def reset(self):
    """
    Resets the environment to its initial state.

    Returns:
        np.ndarray: The initial state vector of the environment.
    """
    # Ball starts near top with random x and downward velocity
    self.ball_x = np.random.uniform(0.1, 0.9)
    self.ball_y = np.random.uniform(0.6, 0.9)
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)  # mostly downward
    speed = np.random.uniform(0.2, 0.5)
    self.ball_vx = speed * np.sin(angle)
    self.ball_vy = -speed * np.cos(angle)  # downward negative

    # Paddle starts at center bottom
    self.paddle_x = 0.5

    self.done = False
    self.reward = 0.0
    return self._get_state()

  def _get_state(self):
    return np.array([self.ball_x, self.ball_y,
                     self.ball_vx, self.ball_vy,
                     self.paddle_x], dtype=np.float32)

  def step(self, action):
    """
    Advances the environment by one time step based on the given action.

    Args:
        action (float): A scalar in [-1, 1] representing the paddle's velocity.

    Returns:
        tuple: (next_state (np.ndarray), reward (float), done (bool))
    """
    if self.done:
      raise RuntimeError("Episode finished. Call reset().")

    # Clip action and move paddle
    vel = np.clip(action, -1.0, 1.0) * self.paddle_speed
    self.paddle_x += vel
    self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

    # Ball physics (simple Euler integration)
    self.ball_x += self.ball_vx * self.dt
    self.ball_y += self.ball_vy * self.dt

    # Bounce off left/right walls
    if self.ball_x < 0:
      self.ball_x = -self.ball_x
      self.ball_vx = abs(self.ball_vx)
    elif self.ball_x > 1:
      self.ball_x = 2 - self.ball_x
      self.ball_vx = -abs(self.ball_vx)

    # Bounce off top wall
    if self.ball_y > 1:
      self.ball_y = 2 - self.ball_y
      self.ball_vy = -abs(self.ball_vy)

    # Check catch or miss at bottom
    reward = -0.1  # step penalty
    if self.ball_y <= 0.0:
      # Check paddle overlap
      left_edge = self.paddle_x - self.paddle_width / 2
      right_edge = self.paddle_x + self.paddle_width / 2
      if left_edge <= self.ball_x <= right_edge:
        reward = 10.0  # successful catch
      else:
        reward = -10.0  # missed
      self.done = True

    self.reward = reward
    return self._get_state(), reward, self.done

  def render(self):
    """Simple console rendering (optional)"""
    grid = [[' ' for _ in range(20)] for _ in range(10)]
    # ball position
    bx = int(self.ball_x * 19)
    by = 9 - int(self.ball_y * 9)
    if 0 <= bx < 20 and 0 <= by < 10:
      grid[by][bx] = 'o'
    # paddle
    px = int(self.paddle_x * 19)
    for i in range(max(0, px - 1), min(20, px + 2)):
      grid[9][i] = '='
    print('\n'.join(''.join(row) for row in grid))
    print(
      f"Ball: ({self.ball_x:.2f},{self.ball_y:.2f}) Paddle: {self.paddle_x:.2f}")

# p = PaddleCatch()

# p.render()
# p.step(0.1)
# p.render()
# p.step(0.2)
# p.render()
# p.step(0.3)
# p.render()
# p.step(0.4)
# p.render()
# p.step(0.5)
# p.render()
# p.step(0.6)
