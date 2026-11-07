import numpy as np


class SumTenEnvironment:
  score_cap = 1000
  directions = ((-1, 0), (0, -1))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.grid = np.zeros((10, 10), dtype=np.uint8)
    self.grid[:, 0] = self.random.integers(1, 10, 10)
    self.grid[6, 1:7] = (6, 5, 8, 1, 7, 3)
    self.x, self.y = 7, 6
    self.scroll_x, self.scroll_y = 100., 0.
    self.direction = self.timer = self.total = self.count = self.steps = 0
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _next(self, direction):
    dx, dy = self.directions[direction]
    return self.x + dx, self.y + dy

  def _value(self, x, y):
    return int(self.grid[y, x]) if 0 <= x < 10 and 0 <= y < 10 else 0

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    self.timer += 1
    if self.timer == 30:
      alternate = 1 - self.direction
      if self._value(*self._next(alternate)):
        self.direction = alternate
      self.timer = 0
    if self.scroll_x - self.x * 11 < 0 or self.scroll_y + self.y * 11 + 1 > 91:
      self.terminal = True
      return 0
    reward = 0
    if tap:
      x, y = self._next(self.direction)
      value = self._value(x, y)
      if not value:
        self.terminal = True
        return 0
      self.x, self.y = x, y
      self.timer = 0
      self.total += value
      self.count += 1
      if self.total % 10 == 0:
        reward = (self.total // 10) ** 2 * 10
        self.total = self.count = 0
      elif self.count == 10:
        self.terminal = True
      if not self._value(*self._next(self.direction)):
        self.direction = 1 - self.direction
    self.scroll_x -= max(0, 7 - self.x) * 0.5 + 0.01
    self.scroll_y += max(0, 6 - self.y) * 0.5 + 0.01
    if self.scroll_x <= 90:
      self.grid[:, 1:] = self.grid[:, :-1].copy()
      self.grid[:, 0] = self.random.integers(1, 10, 10)
      gap = int(self.random.integers(1, 15))
      if gap < 10 and self.grid[gap - 1, 1]:
        self.grid[gap, 0] = 0
      self.scroll_x += 10
      self.x += 1
    if self.scroll_y >= 10:
      self.grid[1:] = self.grid[:-1].copy()
      self.grid[0] = self.random.integers(1, 10, 10)
      gap = int(self.random.integers(1, 15))
      if gap < 10 and self.grid[1, gap - 1]:
        self.grid[0, gap] = 0
      self.scroll_y -= 10
      self.y += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for bit in range(4):
      state[:, :, bit] = (self.grid & (1 << bit)) != 0
    if 0 <= self.x < 10 and 0 <= self.y < 10:
      state[self.y, self.x, 4] = True
    x, y = self._next(self.direction)
    if 0 <= x < 10 and 0 <= y < 10:
      state[y, x, 5] = True
    # Reserve a column for each gauge by using the marker planes' last rows.
    packed = (self.total | (self.count << 7) | (self.timer << 11)
              | (int(self.held) << 16) | (self.direction << 17))
    for bit in range(18):
      state[8 + bit // 10, bit % 10, 5] = bool(packed & (1 << bit))
    return state
