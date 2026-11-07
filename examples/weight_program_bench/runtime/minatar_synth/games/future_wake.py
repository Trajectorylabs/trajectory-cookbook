import math

import numpy as np


class FutureWakeEnvironment:
  score_cap = 2500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.paddle = self.x = self.y = 50.0
    self.vx = self.vy = math.sqrt(0.5)
    self.speed = self.multiplier = 1.0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _position(self, distance: float) -> tuple[float, float, float]:
    offset = (self.x + self.vx * distance - 8.5) % 166
    if offset <= 83:
      x, vx = 8.5 + offset, self.vx
    else:
      x, vx = 174.5 - offset, -self.vx
    return x, self.y + self.vy * distance, vx

  def _impact(self) -> tuple[float, float, float]:
    edge = 91 if self.vy > 0 else 9
    distance = max(0, (edge - self.y) / self.vy)
    x, _, vx = self._position(distance)
    return distance, x, vx

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    direction = -1 if action == 1 else 1 if action == 4 else 0
    reward = 0
    for _ in range(4):
      self.paddle = min(83, max(17, self.paddle + direction * 2))
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    distance, impact_x, impact_vx = self._impact()
    offset = (impact_x - self.paddle) / 10
    ready = abs(offset) <= 1.15
    reward = 0
    if ready and self.speed >= distance:
      self.x = impact_x
      self.y = 91 if self.vy > 0 else 9
      if abs(offset) < 0.6:
        self.vx, self.vy = impact_vx, -self.vy
      else:
        angle = math.pi / 4 + 1 - offset if offset > 0 else 3 * math.pi / 4 - 1 - offset
        self.vx, self.vy = math.cos(angle), math.copysign(math.sin(angle), -self.vy)
      reward = math.floor(self.multiplier)
      self.multiplier += 10
      if self.multiplier > 100:
        self.vx += self.random.uniform(-1, 1) / 3
        length = math.hypot(self.vx, self.vy)
        self.vx /= length
        self.vy /= length
    else:
      self.x, self.y, self.vx = self._position(self.speed)
    self.speed = self.speed + 1 if ready else 1
    self.multiplier = min(100.99, max(1, self.multiplier * 0.99))
    self.terminal = not 0 <= self.y <= 99
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    if 0 <= self.y < 100:
      state[int(self.y) // 10, int(self.x) // 10, 0] = True
    left, right = int(self.paddle - 10) // 10, int(self.paddle + 10) // 10
    state[(0, 9), left:right + 1, 1] = True
    distance, impact_x, _ = self._impact()
    for offset in np.linspace(0, distance, 20):
      x, y, _ = self._position(offset)
      if 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, 2] = True
    state[9 if self.vy > 0 else 0, int(impact_x) // 10, 3] = True
    state[:, (0, 9), 4] = True
    for row, value in enumerate((self.steps, round(self.speed), math.floor(self.multiplier))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
