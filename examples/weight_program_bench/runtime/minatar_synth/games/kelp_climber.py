from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Kelp:
  x: float
  y: float


@dataclass
class Obstacle:
  x: float
  y: float
  width: float
  height: float


class KelpClimberEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = 50.0
    self.kelp = [Kelp(25, 40), Kelp(50, 50), Kelp(75, 45)]
    self.grabbed = self.kelp[1]
    self.obstacles: list[Obstacle] = []
    self.kelp_distance = self.obstacle_distance = 0.0
    self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _nearest(self) -> Kelp | None:
    return min((strand for strand in self.kelp if strand is not self.grabbed and strand.y > 40),
               key=lambda strand: abs(strand.x - self.x), default=None)

  def _advance(self, tap: bool) -> int:
    reward = 0
    if tap:
      target = self._nearest()
      if target is not None:
        reward = math.ceil(abs(target.x - self.x))
        self.grabbed.y = 199
        self.grabbed = target
        self.x = target.x
    for strand in self.kelp:
      strand.y += 0.5
    self.kelp = [strand for strand in self.kelp if strand.y <= 150]
    self.kelp_distance -= 0.5
    if self.kelp_distance < 0:
      self.kelp.append(Kelp(self.random.uniform(10, 90), -50))
      self.kelp_distance += self.random.uniform(30, 50)
    for obstacle in self.obstacles:
      obstacle.y += 0.5
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y <= 110]
    self.obstacle_distance -= 0.5
    if self.obstacle_distance < 0:
      x = self.random.uniform(10, 90)
      width, height = self.random.uniform(5, 15, 2)
      self.obstacles.append(Obstacle(x, -10, width, height))
      self.obstacle_distance += self.random.uniform(40, 50)
    self.terminal = any(abs(o.x - self.x) < (o.width + 5) / 2
                        and abs(o.y - 90) < (o.height + 5) / 2 for o in self.obstacles)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x) // 10, 0] = True
    for strand in self.kelp:
      top = max(0, int((strand.y - 50) // 10))
      bottom = min(9, int((strand.y + 50) // 10))
      state[top:bottom + 1, int(strand.x) // 10, 1 + (strand.y > 40 and strand is not self.grabbed)] = True
    for o in self.obstacles:
      top = max(0, int((o.y - o.height / 2) // 10))
      bottom = min(9, int((o.y + o.height / 2) // 10))
      left = max(0, int((o.x - o.width / 2) // 10))
      right = min(9, int((o.x + o.width / 2) // 10))
      if top <= bottom:
        state[top:bottom + 1, left:right + 1, 3] = True
    target = self._nearest()
    if target is not None:
      state[9, int(target.x) // 10, 4] = True
    for row, value in enumerate((self.steps, round(self.x * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
