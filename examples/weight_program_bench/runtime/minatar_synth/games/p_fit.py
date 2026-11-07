import numpy as np


class PFitEnvironment:
  score_cap = 2400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.cursor = 50
    self.steps = self.action = 0
    self.terminal = False
    self._new_pair()

  def _new_pair(self):
    start = self.random.uniform(22, 32)
    heights = start + np.cumsum(self.random.uniform(10, 50, 6) / 6)
    self.heights = self.random.permutation(heights)
    for _ in range(9):
      self.target = self.random.uniform(10, 90)
      if abs(self.offset) > 20:
        break
    self.fixed_y, self.moving_y = 75.0, 225.0
    self.fixed_target, self.moving_target = -25.0, 125.0
    self.match_ticks = 0
    self.points = 300

  @property
  def offset(self):
    return (self.cursor - self.target + 60) % 120 - 60

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    if self.match_ticks <= 30:
      self.cursor = min(99, max(0, self.cursor + 4 * ((self.action == 3) - (self.action == 1))))
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.fixed_y += (self.fixed_target - self.fixed_y) * 0.1
    self.moving_y += (self.moving_target - self.moving_y) * 0.1
    if self.match_ticks > 30:
      self.match_ticks += 1
      if self.match_ticks == 60:
        self._new_pair()
      return 0
    self.fixed_target += 0.02
    self.moving_target -= 0.02
    if self.match_ticks < 30 and self._collision():
      self.terminal = True
      return 0
    self.match_ticks = self.match_ticks + 1 if abs(self.offset) < 3 else 0
    if self.match_ticks > 30:
      self.fixed_target, self.moving_target = 1, 99
      return self.points
    self.points = max(0, self.points - 1)
    return 0

  def _collision(self):
    for i, height in enumerate(self.heights):
      left = 20 * i - 10
      fixed_bottom = self.fixed_y + height
      for j, moving_height in enumerate(self.heights):
        moving_left = (20 * j - 10 + self.offset + 20) % 120 - 20
        if abs(left - moving_left) < 20 and self.moving_y + moving_height - 99 < fixed_bottom:
          return True
    return False

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    fixed = self.heights[((cols + 10) // 20) % 6]
    moving = self.heights[(((cols - self.offset + 10) % 120) // 20).astype(int)]
    state[:, :, 0] = (rows >= self.fixed_y) & (rows < self.fixed_y + fixed)
    state[:, :, 1] = (rows <= self.moving_y) & (rows > self.moving_y + moving - 99)
    state[:, :, 2] = (rows <= self.moving_y) & (rows > self.moving_y + fixed - 99)
    state[9, self.cursor // 10, 3] = True
    state[9, int(self.target // 10), 4] = True
    values = (self.steps, self.points, self.match_ticks, round((self.offset + 60) * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
