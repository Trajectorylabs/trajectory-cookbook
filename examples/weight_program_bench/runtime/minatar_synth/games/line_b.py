import math

import numpy as np


class LineBEnvironment:
  score_cap = 350

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    angles = self.random.uniform(-math.pi / 3, math.pi / 3, 2) + np.array([0, math.pi])
    self.points = np.full((2, 2), 50.0)
    self.velocity = np.column_stack((np.cos(angles), np.sin(angles)))
    self.gold = None
    self.radius = 1.0
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
    reward = 0
    for frame in range(4):
      reward += self._advance(tapped and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _distance(self, point: np.ndarray) -> float:
    start, end = self.points
    segment = end - start
    length_squared = float(segment @ segment)
    fraction = float((point - start) @ segment) / length_squared if length_squared else 0
    return float(np.linalg.norm(point - (start + min(1, max(0, fraction)) * segment)))

  def _advance(self, tapped: bool) -> int:
    if self.gold is None:
      for _ in range(99):
        self.gold = self.random.uniform(20, 80, 2)
        if self._distance(self.gold) > 22:
          break
      self.radius = 1
    ring = self.radius
    hits_wall = ring + 1.5 >= min(*self.gold - 3, *(97 - self.gold))
    self.radius += 0.05
    if tapped:
      for index, delta in enumerate(self.points.mean(axis=0) - self.points):
        angle = math.atan2(delta[1], delta[0])
        self.velocity[index] += (9 * math.cos(angle), 9 * math.sin(angle))
    self.points += self.velocity
    for point, velocity in zip(self.points, self.velocity, strict=True):
      for axis, low in enumerate((4, 5)):
        if (point[axis] < low and velocity[axis] < 0) or (point[axis] > 96 and velocity[axis] > 0):
          velocity[axis] *= -1
          other = 1 - axis
          jitter = self.random.uniform(0.1, 0.2) * self.random.choice((-1, 1))
          velocity[other] += jitter / (abs(velocity[other]) + 1)
      if np.linalg.norm(velocity) > 1:
        velocity *= 0.9
      velocity *= 0.999
    farthest = max(np.linalg.norm(point - self.gold) for point in self.points)
    if self._distance(self.gold) <= ring + 2 and farthest >= ring - 2:
      self.gold = None
      return math.ceil(self.radius ** 1.5)
    self.terminal = hits_wall
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, (x, y) in enumerate(self.points):
      state[min(9, max(0, int(y // 10))), min(9, max(0, int(x // 10))), channel] = True
    for x, y in np.linspace(*self.points, 30):
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 2] = True
    if self.gold is not None:
      x, y = self.gold
      state[int(y // 10), int(x // 10), 3] = True
      rows, cols = np.ogrid[5:100:10, 5:100:10]
      state[:, :, 4] = abs(np.hypot(cols - x, rows - y) - self.radius) < 6
    for row, value in enumerate((self.steps, round(self.radius * 10))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
