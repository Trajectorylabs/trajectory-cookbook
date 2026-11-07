from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
HALF_PI = math.pi / 2


def wrap(angle: float) -> float:
  return (angle + math.pi) % math.tau - math.pi


@dataclass
class Wall:
  angle: float
  width: float
  y: float
  floating: bool
  bonus: bool

  def projection(self, player_angle: float, rear: bool = False) -> tuple[float, float]:
    relative = player_angle - self.angle + (math.pi if rear else 0)
    start = max(-HALF_PI, min(HALF_PI, wrap(relative - self.width / 2)))
    end = max(-HALF_PI, min(HALF_PI, wrap(relative + self.width / 2)))
    radius = 48 if self.floating else 44
    return 50 + math.sin(start) * radius, 50 + math.sin(end) * radius


class CTowerEnvironment:
  score_cap = 5000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.angle, self.y, self.speed = 0.0, 80.0, 3.0
    self.angular_velocity = -1.0
    self.direction = -1
    self.next_wall = 0.0
    self.next_bonus = self.multiplier = 1
    self.walls: list[Wall] = []

  @property
  def player_x(self) -> float:
    return 50 + 30 * self.angular_velocity

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0.0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance_frame(self) -> float:
    self.speed += ((9 if self.pressed else 1) - self.speed) * 0.2
    self.angular_velocity += (self.direction - self.angular_velocity) * 0.2
    self.y = max(80, self.y - 0.01)
    self.angle = wrap(self.angle + self.angular_velocity * 0.05)
    scroll = self.speed * 0.3
    self.next_wall -= scroll
    if self.next_wall < 0:
      self.next_bonus -= 1
      bonus = self.next_bonus < 0
      if bonus:
        self.next_bonus = int(self.random.integers(9, 13))
      self.walls.append(Wall(self.random.uniform(0, math.tau),
                             self.random.uniform(math.pi / 4, HALF_PI),
                             -5, bonus or self.random.random() < 0.3, bonus))
      self.next_wall += self.random.uniform(30, 40)

    for wall in self.walls:
      wall.y += scroll
      left, right = wall.projection(self.angle)
      if left >= right or not (left - 3 < self.player_x < right + 3 and self.y - 3 < wall.y < self.y + 8):
        continue
      if wall.bonus:
        self.multiplier += 1
        wall.bonus = False
      elif not wall.floating:
        self.angle = wrap(wall.angle - self.direction * (wall.width / 2 + 0.7))
        self.direction *= -1
        self.y = min(99, wall.y + 6)
    self.walls = [wall for wall in self.walls if wall.y <= 105]
    self.terminal = self.y > 97
    return scroll * self.multiplier

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y) // 10, int(self.player_x) // 10, 0] = True
    for wall in self.walls:
      if 0 <= wall.y < 100:
        row = int(wall.y) // 10
        for rear in (False, True):
          left, right = wall.projection(self.angle, rear)
          if left < right:
            channel = 4 if rear else 3 if wall.bonus else 2 if wall.floating else 1
            state[row, int(left) // 10:int(right) // 10 + 1, channel] = True
    for row, value in enumerate((self.steps, self.multiplier, round(self.speed * 10))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
