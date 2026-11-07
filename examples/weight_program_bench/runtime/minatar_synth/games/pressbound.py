import numpy as np


class PressboundEnvironment:
  score_cap = 3000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.pressure = 500
    self.charge = self.steps = 0
    self.held = self.terminal = False
    self._spawn(1, 10)

  def _spawn(self, count, margin):
    self.balls = self.random.uniform(margin, 100 - margin, (count, 2))
    self.velocity = self.random.uniform(0.5, 1, (count, 2))
    self.velocity *= self.random.choice((-1, 1), (count, 2))

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    reward = 0
    self.steps += 1
    for _ in range(4):
      reward += self._advance(held)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, held):
    self.pressure += 20 if held else 5
    reward = 0
    if held:
      if not self.held:
        self.pressure += 200
      self.charge += 1
      margin = self.pressure / 100 + 0.5
      self.terminal = bool(np.any(self.balls < margin) or np.any(self.balls > 100 - margin))
    else:
      if self.held:
        reward = (self.charge * len(self.balls)) ** 2 // 100
        self.charge = 0
        self.pressure = 500
        self._spawn(self.random.integers(2, 5), 5)
      self.terminal = self.pressure >= 4300
      margin = self.pressure / 100 + 1
      self.balls += self.velocity
      self.velocity = np.where(self.balls < margin, abs(self.velocity), self.velocity)
      self.velocity = np.where(self.balls > 100 - margin, -abs(self.velocity), self.velocity)
      np.clip(self.balls, margin, 100 - margin, out=self.balls)
    self.held = held
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    coordinates = np.arange(5, 100, 10)
    edges = np.minimum(coordinates, 100 - coordinates)
    state[:, :, 2] = np.minimum(edges[:, None], edges) <= self.pressure / 100
    for x, y in self.balls:
      row, col = int(y // 10), int(x // 10)
      state[row, col, int(self.held)] = True
      if min(x, y, 100 - x, 100 - y) < self.pressure / 100 + 6:
        state[row, col, 3] = True
    for row, value in enumerate((self.pressure, self.charge, len(self.balls), self.steps)):
      for bit in range(13):
        state[row + bit // 10 * 4, bit % 10, 4] = bool(value & (1 << bit))
    state[:, :, 5] = not self.held and self.pressure > 3500
    return state
