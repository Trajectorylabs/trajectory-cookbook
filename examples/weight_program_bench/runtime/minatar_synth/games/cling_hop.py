from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Platform:
  x: float
  y: float
  width: float
  phase: float
  identity: int

  def center(self, frame: int) -> float:
    return min(98 - self.width / 2, max(self.width / 2 + 2,
               self.x + 20 * math.sin(frame * 0.05 + self.phase)))


@dataclass
class Obstacle:
  x: float
  y: float
  vx: float
  vy: float

  def advance(self) -> None:
    self.x += self.vx
    self.y += self.vy
    if not 3 <= self.x <= 97:
      self.x = min(97, max(3, self.x))
      self.vx *= -1
    if not 3 <= self.y <= 97:
      self.y = min(97, max(3, self.y))
      self.vy *= -1


class ClingHopEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = self.obstacle_timer = 0
    self.pressed = self.terminal = False
    self.x = self.y = 50.0
    self.vy = 0.0
    self.support: Platform | None = None
    self.last_platform = -1
    self.platforms: list[Platform] = []
    self.obstacles: list[Obstacle] = []

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
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, flap: bool) -> int:
    if self.frame % 90 == 0:
      self.platforms.append(Platform(self.random.uniform(20, 80), -5,
                                    self.random.uniform(15, 30), self.random.uniform(0, math.tau),
                                    self.frame))
    self.obstacle_timer -= 1
    if self.obstacle_timer < 0 or not self.obstacles:
      direction = 1 if self.x > 50 else -1
      self.obstacles.append(Obstacle(3 if direction > 0 else 97, self.random.uniform(20, 60),
                                    direction * self.random.uniform(0.5, 1.2), self.random.uniform(-1, 1)))
      self.obstacle_timer = 180

    reward = 0
    if self.support is None:
      self.vy = -2.2 if flap else self.vy + 0.12
      self.y += self.vy
    else:
      self.x = self.support.center(self.frame)
      self.y = self.support.y + 7
      self.vy = 0
      reward += 1
      if flap:
        self.support = None
        self.vy = -2.5
    self.terminal = self.y > 105

    for platform in self.platforms:
      platform.y += 0.3
      if (self.support is None and self.vy > 0 and -3 < self.y - platform.y < 8
          and abs(self.x - platform.center(self.frame)) < platform.width / 2 + 3):
        self.support = platform
        if platform.identity != self.last_platform:
          self.last_platform = platform.identity
          reward += len(self.obstacles)
    self.platforms = [platform for platform in self.platforms if platform.y < 110]
    if self.support is not None and self.support.y >= 110:
      self.support = None

    survivors = []
    for obstacle in self.obstacles:
      obstacle.advance()
      if abs(obstacle.x - self.x) < 5 and abs(obstacle.y - self.y) < 5:
        if self.support is not None:
          self.terminal = True
          survivors.append(obstacle)
      else:
        survivors.append(obstacle)
    self.obstacles = survivors
    self.frame += 1
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = min(9, max(0, int(self.y) // 10)), int(self.x) // 10
    state[row, col, 0] = True
    state[row, col, 3] = self.support is not None
    state[row, col, 4] = self.vy < 0
    for platform in self.platforms:
      if 0 <= platform.y < 100:
        x = platform.center(self.frame)
        left = int(x - platform.width / 2) // 10
        right = int(x + platform.width / 2) // 10
        state[int(platform.y) // 10, left:right + 1, 1] = True
    for obstacle in self.obstacles:
      state[int(obstacle.y) // 10, int(obstacle.x) // 10, 2] = True
    for row, value in enumerate((self.steps, len(self.obstacles), self.obstacle_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
