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
  - Gravity: The bird naturally falls by 1 row every 2 ticks (odd ticks) if it does not flap.
  - Actions:
    - 0 (NOTHING): Let gravity take effect (bird drifts down 1 row every 2 ticks).
    - 1 (FLAP): The bird jumps up 1 row.
    - 2 (FLAP_FLAP): The bird jumps up 2 rows.
  - Boundaries:
    - Floor: Reaching the bottom boundary (row H-1) results in immediate episode termination (death).
    - Ceiling: Bumping into the ceiling (row 0) clamps the bird to row 0 and gives a small penalty.

  Collision Detection:
  - Checked at the start of each tick before the bird takes its action, ensuring that
    wall entry and collisions are rendered accurately in column 0.

  Rewards:
  - +1 for successfully navigating a wall (when a wall passes column 0 without collision).
  - -10 for hitting a wall or falling to the floor.
  - -1 for hitting the ceiling.

  State Representation:
  The state space is designed to be fully Markovian while keeping the size small:
  `state = (bird_pos, distance_to_wall, closest_wall_end_row, second_wall_end_row, gravity_phase)`
  - bird_pos: vertical coordinate of the bird (0 to H-1)
  - distance_to_wall: horizontal coordinate of the closest wall (0 to W-1)
  - closest_wall_end_row: vertical coordinate of the bottom end of the closest wall gap (0 to H-1)
  - second_wall_end_row: vertical coordinate of the bottom end of the second wall gap (0 to H-1)
  - gravity_phase: binary indicator of whether gravity will trigger next tick (0 or 1)
  """

  def __init__(self, dim=(15, 25)):
    self.rc = dim
    self.H, self.W = self.rc
    self.grid_height = self.H
    self.grid_width = self.W
    self.reset()

  def get_sorted_walls(self):
    in_front = [w for w in self.walls if w['column'] >= 0]
    return sorted(in_front, key=lambda w: w['column'])

  def pos_to_state(self):
    bird_pos = self.bird_vertical_pos
    sorted_walls = self.get_sorted_walls()
    closest_wall = sorted_walls[0]
    second_wall = sorted_walls[1]
    distance_to_wall = closest_wall['column']
    wall_end_row = (closest_wall['row_start'] +
                    closest_wall['height']) % self.grid_height
    second_wall_end_row = (second_wall['row_start'] +
                           second_wall['height']) % self.grid_height
    gravity_phase = self.tick_count & 1
    # H x W x H x H x 2
    return bird_pos, distance_to_wall, wall_end_row, second_wall_end_row, gravity_phase

  def check_wall_collides(self, wall):
    for i in range(wall['height']):
      row = (wall['row_start'] + i) % self.grid_height
      if self.bird_vertical_pos == row:
        return True
    return False

  def tick(self, action):
    reward = 0

    # Check wall collision FIRST, before bird moves
    for wall in self.walls:
      if wall['column'] == 0:
        if self.check_wall_collides(wall):
          reward = -10
          self.done = True
          return self.pos_to_state(), reward, self.done
        else:
          reward = 1
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
    if self.tick_count & 1 and not flap:
      self.bird_vertical_pos += 1

    # Check grid boundaries
    if self.bird_vertical_pos >= self.grid_height - 1:
      self.done = True
      reward = -10
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
    ]

    self.done = False
    self.tick_count = 0
    self.bird_vertical_pos = self.walls[0]['height'] >> 1
    return self.pos_to_state()

  def render(self):
    grid = np.full((self.grid_height, self.grid_width), '   ')
    r, c = self.bird_vertical_pos, 0

    for wall in self.walls:
      wc = wall['column']
      if 0 <= wc < self.grid_width:
        for i in range(wall['height']):
          wr = wall['row_start']
          p = (wr + i) % self.grid_height
          grid[(p, wc)] = ' █ '

    grid[(r, c)] = ' 🦉'

    print('\n'.join([''.join(row) for row in grid]))
    print()
