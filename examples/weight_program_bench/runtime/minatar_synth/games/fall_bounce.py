from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Platform:
  x: float
  y: float
  width: float


class FallBounceEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.pressed = self.bouncing = self.terminal = False
    self.x, self.y = 50.0, 10.0
    self.platforms: list[Platform] = []
    self._spawn_pair()

  def _spawn_pair(self) -> None:
    for _ in range(2):
      self.platforms.append(Platform(self.random.uniform(0, 85), -5, self.random.uniform(15, 35)))

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
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self) -> float:
    if self.bouncing:
      self.y += (10 - self.y) * 0.2
      self.bouncing = abs(10 - self.y) >= 1
    else:
      self.y += 1.5 if self.pressed else 0.25
    self.x = (self.x + 1) % 100
    if self.y > 103:
      self.terminal = True
      return 0.0

    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self._spawn_pair()
      self.spawn_timer = math.ceil(self.random.uniform(40, 60))
    reward = 0.0
    for platform in self.platforms:
      platform.y += 1
      if (self.pressed and not self.bouncing and abs(platform.y - self.y) < 4
          and abs(platform.x - self.x) < platform.width / 2 + 5):
        self.bouncing = True
        reward = self.y - 10
    self.platforms = [platform for platform in self.platforms if platform.y < 103]
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = min(9, int(self.y) // 10), int(self.x) // 10
    state[row, col, 0] = True
    state[row, col, 2] = self.bouncing
    state[row, col, 3] = self.pressed
    for platform in self.platforms:
      if 0 <= platform.y < 100:
        left = max(0, math.floor((platform.x - platform.width / 2) / 10))
        right = min(10, math.ceil((platform.x + platform.width / 2) / 10))
        state[int(platform.y) // 10, left:right, 1] = True
    state[9, :, 4] = True
    for row, value in enumerate((self.steps, self.spawn_timer, round(self.y * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
