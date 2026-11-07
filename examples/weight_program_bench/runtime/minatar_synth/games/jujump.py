import numpy as np


class JujumpEnvironment:
  score_cap = 600

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 50.0
    self.vx = self.vy = 0.0
    self.direction = 1
    self.power = self.spawn_distance = 1.0
    self.platforms = [(50.0, 70.0)]
    self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tapped = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for frame in range(4):
      reward += self._advance(tapped and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, tapped: bool) -> float:
    self.x += self.vx
    self.y += self.vy
    self.vy += 0.05 if self.pressed else 0.1
    scroll = max(0, 30 - self.y) * 0.1 + 0.1
    self.spawn_distance -= scroll
    if self.spawn_distance < 0:
      self.spawn_distance = self.random.uniform(0, 99)
      self.platforms.append((self.random.uniform(0, 99), -9))
    self.y += scroll
    self.platforms = [(x, y + scroll) for x, y in self.platforms if y + scroll < 99]
    while any(abs(self.x - x) < 20 and abs(self.y - y) < 7 for x, y in self.platforms):
      self.y -= 1
      self.vx = self.vy = 0
      self.power = 1
    if tapped:
      self.direction *= -1
      self.vx = self.direction
      self.vy = -3 * self.power
      self.power *= 0.7
    self.terminal = self.y > 99
    return scroll

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for x, y, channel in ((self.x, self.y, 0), (self.x + 4 * self.vx, self.y + 4 * self.vy, 2),
                          (self.x - 10 * self.direction, self.y - 10 * self.power, 3)):
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, channel] = True
    for x, y in self.platforms:
      if 0 <= y < 100:
        left = max(0, int((x - 16.5) // 10))
        right = min(9, int((x + 16.5) // 10))
        state[int(y) // 10, left:right + 1, 1] = True
    state[3, :, 4] = True
    values = (self.steps, round(self.power * 400), min(511, max(0, round((self.vy + 3) * 40))))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
