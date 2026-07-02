import random
import numpy as np

NOTHING = 0
FLAP = 1
FLAP_FLAP = 2

A = {
    NOTHING: '~',
    FLAP: "flap",
    FLAP_FLAP: "flap flap"
}


class FlappyGrid:
  """
  FlappyGrid is a grid-based reinforcement learning environment inspired by Flappy Bird.

  The environment consists of a 2D grid of size H x W.
  The bird (represented by '🦉') sits at column 0 and can fly vertically.
  Obstacles (walls, represented by '█') drift from right to left at a rate of 1 column per tick,
  while also drifting vertically downwards.

  Grid Properties:
  - Vertical wrapping: The walls wrap around vertically, leaving a gap for the bird to navigate through.
  - Two active walls are present on the grid, spaced evenly apart to ensure the environment
    is continuously dynamic.

  Movement & Physics:
  - Gravity: The bird naturally falls by 1 row every tick if it does not flap.
  - Actions:
    - 0 (NOTHING): Let gravity take effect (bird drifts down 1 row).
    - 1 (FLAP): The bird jumps up 1 row.
    - 2 (FLAP_FLAP): The bird jumps up 2 rows.
  - Boundaries:
    - Floor: Reaching the bottom boundary (row H-1) results in immediate episode termination (death).
    - Ceiling: Bumping into the ceiling (row 0) clamps the bird to row 0 and gives a small penalty.

  Collision Detection:
  - Checked at the start of each tick before the bird takes its action, ensuring that
    wall entry and collisions are rendered accurately in column 0.

  Rewards:
  - +5 for successfully navigating a wall (when a wall passes column 0 without collision).
  - -15 for hitting a wall or falling to the floor.
  - -1 for hitting the ceiling.
  """

  def __init__(self, dim=(15, 30)):
    self.rc = dim
    self.H, self.W = self.rc
    self.grid_height = self.H
    self.grid_width = self.W
    self.reset()

  def get_sorted_walls(self):
    in_front = [w for w in self.walls if w['column'] >= 0]
    return sorted(in_front, key=lambda w: w['column'])

  def pos_to_state(self):
    """
    Helper function to convert a FlappyGrid instance into a flattened
    numerical grid representation.

    Empty spaces are represented as 0.0.
    Wall segments are represented as 1.0.
    The bird is represented as 0.5.

    Returns:
        np.ndarray: Flattened array of shape (grid_height * grid_width,)
    """
    grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

    # Draw walls
    for wall in self.walls:
      wc = wall['column']
      if 0 <= wc < self.grid_width:
        for i in range(wall['height']):
          wr = wall['row_start']
          p = (wr + i) % self.grid_height
          grid[p, wc] = 1.0

    # Draw bird
    if 0 <= self.bird_vertical_pos < self.grid_height:
      grid[self.bird_vertical_pos, 0] = 0.5

    return grid.flatten()

  def check_wall_collides(self, wall):
    for i in range(wall['height']):
      row = (wall['row_start'] + i) % self.grid_height
      if self.bird_vertical_pos == row:
        return True
    return False

  def tick(self, action=NOTHING):
    # reward = 0
    reward = 0.01

    # Check wall collision FIRST, before bird moves
    for wall in self.walls:
      if wall['column'] == 0:
        if self.check_wall_collides(wall):
          reward = -15
          self.done = True
          return self.pos_to_state(), reward, self.done
        else:
          reward = 5
        self.reset_wall(wall)

    # Apply action
    flap = action >= FLAP
    if self.bird_vertical_pos > 0:
      if action == FLAP:
        self.bird_vertical_pos -= 1
      if action == FLAP_FLAP:
        self.bird_vertical_pos -= 2
      self.bird_vertical_pos = max(self.bird_vertical_pos, 0)

    # Gravity: after every 2 steps, bird falls by one step
    if not flap:
      self.bird_vertical_pos += 1

    # Check grid boundaries
    if self.bird_vertical_pos >= self.grid_height - 1:
      self.done = True
      reward = -15
      self.bird_vertical_pos = self.grid_height - 1
      return self.pos_to_state(), reward, self.done

    if self.bird_vertical_pos <= 0:
      self.bird_vertical_pos = 0
      reward = -1  # gentle nudge for hitting ceiling

    # Move all walls closer to bird and push them down
    for wall in self.walls:
      if wall['column'] > 0:
        wall['column'] -= 1
        wall['row_start'] += 1

    self.tick_count += 1
    return self.pos_to_state(), reward, self.done

  def reset_wall(self, wall):
    wall['height'] = round(0.7 * self.grid_height)
    wall['row_start'] = random.randint(0, self.grid_height - 1)
    wall['column'] = self.grid_width - 1

  def reset(self):
    def new_wall(column): return {
        'height': round(0.5 * self.grid_height),
        'row_start': random.randint(0, self.grid_height - 1),
        'column': column
    }

    self.walls = [
        new_wall(column=self.grid_width - 1),
        new_wall(column=(self.grid_width - 1) // 2),
        # new_wall(column=(self.grid_width - 1) // 2 - 1),
    ]

    self.done = False
    self.tick_count = 0
    self.bird_vertical_pos = self.walls[0]['height'] >> 1
    return self.pos_to_state()

  def render(self):
    EMPTY = '   '
    grid = np.full((self.grid_height, self.grid_width), EMPTY)
    r, c = self.bird_vertical_pos, 0

    for wall in self.walls:
      wc = wall['column']
      if 0 <= wc < self.grid_width:
        for i in range(wall['height']):
          wr = wall['row_start']
          p = (wr + i) % self.grid_height
          grid[(p, wc)] = ' ██ '

    if grid[(r, c)] == EMPTY:
      grid[(r, c)] = ' 🦉'
    else:
      grid[(r, c)] = ' 🟥 '

    row_width = self.grid_width * 3  # each cell is 3 chars
    top = ' ' + '_' * (row_width + 2)
    bottom = ' ' + '‾' * (row_width + 2)
    lines = [top]
    for row in grid:
      lines.append('|' + ''.join(row) + '  |')
    lines.append(bottom)
    print('\n'.join(lines))
