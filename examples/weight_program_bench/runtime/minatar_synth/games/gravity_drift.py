from dataclasses import dataclass

import numpy as np


@dataclass
class Object:
  x: float
  y: float
  size: float


class GravityDriftEnvironment:
  score_cap = 4000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y, self.vy = 20.0, 50.0, 0.0
    self.steps = self.frames = 0
    self.multiplier = 1
    self.reversed = self.pressed = self.terminal = False
    self.stars: list[Object] = []
    self.asteroids: list[Object] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.x += ((80 if self.reversed else 20) - self.x) * 0.05
    self.vy = (self.vy + 0.12 - 0.25 * self.pressed) * 0.95
    self.y += self.vy
    if not 0 <= self.y <= 100:
      self.terminal = True
      return 0
    spawn_x = -5 if self.reversed else 105
    if self.frames % 25 == 0:
      target = min(100, max(0, self.y + self.random.uniform(-9, 9)))
      self.asteroids.append(Object(spawn_x, target, self.random.uniform(5, 10)))
    if self.frames % 120 == 0:
      self.stars.append(Object(spawn_x, self.random.uniform(20, 80), 4))
    self.frames += 1
    remaining = []
    for star in self.stars:
      star.x += 0.5 if self.reversed else -0.5
      if self._hits_player(star):
        self.reversed = not self.reversed
        self.multiplier += 1
      elif -10 < star.x < 110:
        remaining.append(star)
    self.stars = remaining
    for asteroid in self.asteroids:
      asteroid.x += 1 if self.reversed else -1
      if self._hits_player(asteroid):
        self.terminal = True
    self.asteroids = [asteroid for asteroid in self.asteroids if -10 < asteroid.x < 110]
    return 0 if self.terminal else self.multiplier

  def _hits_player(self, obj: Object) -> bool:
    extent = obj.size / 2 + 2.5
    return abs(obj.x - self.x) < extent and abs(obj.y - self.y) < extent

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), int(self.x) // 10, 0] = True
    for channel, objects in ((1, self.asteroids), (2, self.stars)):
      for obj in objects:
        if 0 <= obj.x < 100:
          state[min(9, int(obj.y) // 10), int(obj.x) // 10, channel] = True
    state[:, 8 if self.reversed else 2, 3] = True
    state[min(9, max(0, int(self.y + 4 * self.vy) // 10)), int(self.x) // 10, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, round((self.vy + 3) * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.reversed
    return state
