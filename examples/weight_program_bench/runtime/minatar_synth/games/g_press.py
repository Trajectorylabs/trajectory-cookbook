from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Bubble:
  x: float
  y: float
  size: float
  velocity: float


@dataclass
class Drop:
  x: float
  y: float
  velocity: float = 0


class GPressEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.drop_timer = 0
    self.pressed = self.terminal = False
    self.y, self.direction, self.width, self.motion = 30, 1, 80.0, 0.0
    self.water = self.target = self.previous = 70.0
    self.bubble_timer = 0.0
    self.bubbles: list[Bubble] = []
    self.drops: list[Drop] = []

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

  def _advance(self, tap: bool) -> int:
    if self.motion:
      self.width += 5 * self.motion
      if self.width < 0:
        self.width, self.motion = 0, 0.5
      elif self.width > 80:
        self.width, self.motion = 80, 0
    else:
      self.y += self.direction
      if (self.y < 10 and self.direction < 0) or (self.y > self.water - 10 and self.direction > 0):
        self.direction *= -1
      if tap:
        self.previous = self.target = self.water + 12
        self.motion = -1

    self.bubble_timer -= 1
    if self.bubble_timer < 0:
      size = self.random.uniform(5, 9)
      self.bubbles.append(Bubble(self.random.uniform(20 + size, 80 - size), self.water + size / 2,
                                 size, -self.random.uniform(1, 2) / size))
      self.target += size * 0.03
      self.bubble_timer += self.random.uniform(10, 50)
    self._move_bubbles()
    reward = 0
    survivors = []
    for drop in self.drops:
      drop.velocity += 0.1
      drop.y += drop.velocity
      if drop.y > self.water + 1:
        self.drop_timer = 9
        self.target -= 0.35
      else:
        survivors.append(drop)
    self.drops = survivors
    if self.drop_timer:
      self.drop_timer -= 1
      if not self.drop_timer and self.previous > self.target:
        gain = self.previous - self.target
        reward = math.ceil(gain**1.5)
    self.target += max(0, 50 - self.target) * 0.05
    self.water += (self.target - self.water) * 0.2
    self.terminal = self.motion == 0 and not self.drops and min(self.water, self.target) >= 100
    return reward

  def _move_bubbles(self) -> None:
    survivors = []
    left, right = 50 - self.width / 2, 50 + self.width / 2
    for bubble in self.bubbles:
      bubble.y += bubble.velocity
      radius = bubble.size / 2
      jaw = left if bubble.x < 50 else right
      if (self.motion < 0 or self.width == 0) and abs(bubble.y - self.y) < 10 + radius and abs(bubble.x - jaw) <= 5 + radius:
        if bubble.size > self.width:
          for _ in range(math.ceil(bubble.size)):
            self.drops.append(Drop(bubble.x + self.random.uniform(-radius, radius),
                                   bubble.y + self.random.uniform(-radius, radius)))
          continue
        bubble.x = jaw + (radius if bubble.x < 50 else -radius)
      if bubble.y >= -radius:
        survivors.append(bubble)
    self.bubbles = survivors

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    top = max(0, (self.y - 10) // 10)
    bottom = min(10, (self.y + 10) // 10 + 1)
    for x in (50 - self.width / 2, 50 + self.width / 2):
      state[top:bottom, int(x) // 10, 0] = True
    for objects, channel in ((self.bubbles, 1), (self.drops, 2)):
      for obj in objects:
        if 0 <= obj.y < 100:
          state[int(obj.y) // 10, int(obj.x) // 10, channel] = True
    state[min(10, int(self.water) // 10):, :, 3] = True
    state[top:bottom, :, 4] = self.motion < 0
    for row, value in enumerate((self.steps, round(self.width), round(self.target), self.drop_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 5] = self.pressed, self.direction > 0, self.motion > 0
    return state
