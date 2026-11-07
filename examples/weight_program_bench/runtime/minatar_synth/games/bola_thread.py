from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
SCROLL = 0.55
ROTATION = 0.022


@dataclass
class Wall:
  x: float
  gap_y: float
  height: float
  nub: bool
  passed: bool = False

  def blocked(self, y: float, margin: float = 0) -> bool:
    return (
      abs(y - self.gap_y) + margin > self.height / 2
      or self.nub and abs(y - 50) < 4 + margin
    )


@dataclass
class Beacon:
  x: float
  y: float


class BolaThreadEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.freeze = 0
    self.pressed = self.terminal = False
    self.angle, self.radius = 0.0, 20.0
    self.next_wall, self.next_beacon = 60.0, 100
    self.walls: list[Wall] = []
    self.beacons: list[Beacon] = []

  @property
  def balls(self) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = math.cos(self.angle) * self.radius
    dy = math.sin(self.angle) * self.radius
    return (30 + dx, 50 + dy), (30 - dx, 50 - dy)

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
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    if self.freeze:
      self.freeze -= 1
      return 0
    self.radius = max(6, self.radius - 0.6) if self.pressed else min(26, self.radius + 0.45)
    self.angle = (self.angle + ROTATION) % math.tau
    self.next_wall -= SCROLL
    self.next_beacon -= 1
    if self.next_wall < 0:
      arrival_angle = self.angle + ROTATION * (108 - 30) / SCROLL
      nub = self.random.random() < 0.36 and abs(math.sin(arrival_angle)) > 0.8
      gap_y = self.random.uniform(42, 58)
      height = self.random.uniform(42, 54) if nub else self.random.uniform(25, 40)
      self.walls.append(Wall(108, gap_y, height, nub))
      self.next_wall = self.random.uniform(65, 90)
    if self.next_beacon < 0:
      self.beacons.append(Beacon(108, self.random.uniform(20, 80)))
      self.next_beacon = int(self.random.integers(90, 151))

    reward = 0
    balls = self.balls
    for wall in self.walls:
      wall.x -= SCROLL
      if any(abs(x - wall.x) < 4 and wall.blocked(y, margin=2) for x, y in balls):
        self.terminal = True
        return reward
      if not wall.passed and wall.x < 30:
        wall.passed = True
        bonus = max(1, int((self.radius - 6) // 3))
        reward += bonus
        self.freeze = 2 + 2 * (bonus >= 3) + 2 * (bonus >= 5)
    self.walls = [wall for wall in self.walls if wall.x >= -5]
    survivors = []
    for beacon in self.beacons:
      beacon.x -= SCROLL
      if any(abs(x - beacon.x) < 4.5 and abs(y - beacon.y) < 4.5 for x, y in balls):
        reward += 8
      elif beacon.x >= -5:
        survivors.append(beacon)
    self.beacons = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 3, 0] = True
    for x, y in self.balls:
      state[int(y) // 10, int(x) // 10, 1] = True
    for wall in self.walls:
      if 0 <= wall.x < 100:
        column = int(wall.x) // 10
        for row in range(10):
          state[row, column, 2] |= abs(10 * row + 5 - wall.gap_y) > wall.height / 2
        state[4:6, column, 3] |= wall.nub
    for beacon in self.beacons:
      if 0 <= beacon.x < 100:
        state[int(beacon.y) // 10, int(beacon.x) // 10, 4] = True
    for row, value in enumerate((self.steps, round(self.radius * 10), round(self.angle * 80), self.freeze)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
