import numpy as np


class InterspaceEnvironment:
  score_cap = 11840

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 10.0
    self.height = 7.0
    self.gap_x, self.wall_y = 50.0, 99.0
    self.steps = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0.0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self) -> float:
    scroll = 0.5 + (self.height - 7) * 0.3
    self.wall_y -= scroll
    if self.wall_y < 0:
      self.gap_x = (self.gap_x + self.random.uniform(-9, 9) - 9) % 81 + 9
      self.wall_y = 99
    self.x = min(99, max(0, self.x + 1.25 * ((self.action == 3) - (self.action == 1))))
    outside_gap = abs(self.x - self.gap_x) > 3.5
    wall_in_probe = self.wall_y < self.y + 49.5 and self.wall_y + 9 > self.y - 49.5
    if outside_gap and wall_in_probe:
      self.y += (10 - self.y) * 0.1
      self.height += (7 - self.height) * 0.4
    else:
      self.y += (30 - self.y) * 0.1
      self.height += (30 - self.height) * 0.1
    self.terminal = (outside_gap and self.y + self.height / 2 > self.wall_y
                     and self.y - self.height / 2 < self.wall_y + 9)
    return scroll

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    col = int(self.x // 10)
    top = max(0, int((self.y - self.height / 2) // 10))
    bottom = min(9, int((self.y + self.height / 2) // 10))
    state[top:bottom + 1, col, 0] = True
    wall_top = int(self.wall_y // 10)
    wall_bottom = min(9, int((self.wall_y + 9) // 10))
    for column in range(10):
      cell_x = column * 10 + 5
      state[wall_top:wall_bottom + 1, column, 1] = abs(cell_x - self.gap_x) >= 7
    state[wall_top:wall_bottom + 1, int(self.gap_x // 10), 2] = True
    state[:, col, 3] = True
    state[:, int(self.gap_x // 10), 4] = True
    gauges = (self.steps, round(self.x * 4), round(self.gap_x * 4),
              round(self.wall_y * 4), round(self.height * 10), round(self.y * 10), self.action)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
