import numpy as np


class PaddleCatchV2:
  """
  A complex continuous-action environment: juggle multiple bouncing balls with a paddle.
  Features wind, gravity, and static obstacles.

  State (2 + 4 * num_balls dim vector):
      - paddle_x ∈ [0, 1] (center of paddle)
      - wind_x ∈ [-1, 1] (current horizontal wind acceleration)
      - For each ball:
          - ball_x, ball_y ∈ [0, 1]
          - ball_vx, ball_vy 

  Action: scalar in [-1, 1] representing paddle velocity.
      - Positive moves paddle right, negative moves left.

  Termination:
      - Reaches max_steps (e.g. 1000). Juggling is continuous.

  Rewards:
      - Ball caught (y <= 0 and paddle overlap) -> +10
      - Ball missed (y <= 0, no overlap) -> -10
      - Small step penalty -> -0.1
  """

  def __init__(self, num_balls=2, paddle_width=0.2, paddle_speed=0.03, dt=0.05, max_steps=1000):
    self.num_balls = num_balls
    self.paddle_width = paddle_width
    self.paddle_speed = paddle_speed
    self.dt = dt
    self.max_steps = max_steps

    # Physics constants
    self.gravity = -0.5
    self.wind_volatility = 0.1

    # Static obstacles: [x_min, x_max, y_min, y_max]
    self.obstacles = [
        [0.1, 0.3, 0.5, 0.55],
        [0.4, 0.6, 0.7, 0.75],
        [0.7, 0.9, 0.5, 0.55]
    ]

    self.reset()

  def reset(self):
    self.steps = 0
    self.paddle_x = 0.5
    self.wind_x = 0.0

    self.balls = []
    for _ in range(self.num_balls):
      self._spawn_ball()

    self.done = False
    self.episode_reward = 0.0
    return self._get_state()

  def _spawn_ball(self):
    x = np.random.uniform(0.1, 0.9)
    y = np.random.uniform(0.8, 1.0)
    vx = np.random.uniform(-0.3, 0.3)
    vy = np.random.uniform(-0.1, -0.3)
    self.balls.append({'x': x, 'y': y, 'vx': vx, 'vy': vy})

  def _get_state(self):
    state = [self.paddle_x, self.wind_x]
    for b in self.balls:
      state.extend([b['x'], b['y'], b['vx'], b['vy']])
    return np.array(state, dtype=np.float32)

  def step(self, action):
    if self.done:
      raise RuntimeError("Episode finished. Call reset().")

    self.steps += 1
    reward = -0.1  # step penalty

    # Update wind (random walk bounded by [-1, 1])
    self.wind_x += np.random.normal(0, self.wind_volatility)
    self.wind_x = np.clip(self.wind_x, -1.0, 1.0)

    # Move paddle
    vel = np.clip(action, -1.0, 1.0) * self.paddle_speed
    self.paddle_x += vel
    self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

    # Process each ball
    for i, b in enumerate(self.balls):
      # Apply forces (gravity and wind)
      b['vy'] += self.gravity * self.dt
      b['vx'] += self.wind_x * self.dt

      # Update position
      next_x = b['x'] + b['vx'] * self.dt
      next_y = b['y'] + b['vy'] * self.dt

      # Wall collisions (Left/Right)
      if next_x < 0:
        next_x = -next_x
        b['vx'] = abs(b['vx'])
      elif next_x > 1:
        next_x = 2 - next_x
        b['vx'] = -abs(b['vx'])

      # Wall collision (Top)
      if next_y > 1:
        next_y = 2 - next_y
        b['vy'] = -abs(b['vy'])

      # Obstacle collisions (simple AABB)
      for obs in self.obstacles:
        if obs[0] <= next_x <= obs[1] and obs[2] <= next_y <= obs[3]:
          # Determine bounce direction based on previous position and reflect distance
          if b['x'] < obs[0]:
            next_x = obs[0] - (next_x - obs[0])
            b['vx'] *= -1
          elif b['x'] > obs[1]:
            next_x = obs[1] + (obs[1] - next_x)
            b['vx'] *= -1

          if b['y'] < obs[2]:
            next_y = obs[2] - (next_y - obs[2])
            b['vy'] *= -1
          elif b['y'] > obs[3]:
            next_y = obs[3] + (obs[3] - next_y)
            b['vy'] *= -1

      b['x'] = next_x
      b['y'] = next_y

      # Check floor / catch
      if b['y'] <= 0.0:
        left_edge = self.paddle_x - self.paddle_width / 2
        right_edge = self.paddle_x + self.paddle_width / 2

        if left_edge <= b['x'] <= right_edge:
          reward += 10.0  # Catch
        else:
          reward -= 10.0  # Miss

        # Respawn the ball at the top to keep juggling
        b['x'] = np.random.uniform(0.1, 0.9)
        b['y'] = 1.0
        b['vx'] = np.random.uniform(-0.3, 0.3)
        b['vy'] = np.random.uniform(-0.1, -0.3)

    if self.steps >= self.max_steps:
      self.done = True

    self.episode_reward += reward
    return self._get_state(), reward, self.done

  def render(self):
    """Higher resolution console rendering (40x20)"""
    W, H = 40, 20
    grid = [[' ' for _ in range(W)] for _ in range(H)]

    # Draw obstacles
    for obs in self.obstacles:
      x_start = int(obs[0] * (W - 1))
      x_end = int(obs[1] * (W - 1))
      y_start = int((1 - obs[3]) * (H - 1))
      y_end = int((1 - obs[2]) * (H - 1))
      for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
          if 0 <= y < H and 0 <= x < W:
            grid[y][x] = '#'

    # Draw balls
    for i, b in enumerate(self.balls):
      bx = int(b['x'] * (W - 1))
      by = int((1 - b['y']) * (H - 1))
      if 0 <= bx < W and 0 <= by < H:
        grid[by][bx] = str(i)

    # Draw paddle
    px = int(self.paddle_x * (W - 1))
    pw = int(self.paddle_width * (W - 1))
    half_w = pw // 2
    for i in range(max(0, px - half_w), min(W, px + half_w + 1)):
      grid[H - 1][i] = '='

    print('\n' + '=' * W)
    print('\n'.join(''.join(row) for row in grid))
    print(f"Wind: {self.wind_x:+.2f} | Paddle: {self.paddle_x:.2f} | Step: {self.steps}/{self.max_steps} | Reward: {self.episode_reward:.2f}")


if __name__ == "__main__":
  import time
  env = PaddleCatchV2()
  state = env.reset()

  total_reward = 0
  for _ in range(1000):
    # Move paddle somewhat randomly
    action = np.random.uniform(-1, 1)
    state, reward, done = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.05)
    if done:
      break
  print(f"Total reward: {total_reward:.2f}")
