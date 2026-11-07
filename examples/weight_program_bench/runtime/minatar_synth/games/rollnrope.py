from dataclasses import dataclass
from math import cos, floor, pi, sin

import numpy as np


@dataclass
class Rope:
  x: float
  angle: float
  radius: float
  size: float
  speed: float
  count: int

  def position(self):
    return (self.x + cos(self.angle) * self.radius,
            91 - self.size / 2 - self.radius + sin(self.angle) * self.radius)


class RollnropeEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vy = 5.0, 84.0, 0.0
    self.mode = "stop"
    self.steps = self.frame = 0
    self.held = self.terminal = False
    self.rope = None

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    if self.rope is None:
      self.rope = Rope(150, self.random.uniform(0, 2 * pi), self.random.uniform(30, 45),
                       self.random.uniform(5, 10), self.random.uniform(1, 2),
                       int(self.random.integers(2, 6)) if self.frame else 2)
      if self.random.random() < 0.2:
        self.rope.speed *= -1
    rope = self.rope
    rope.x -= (rope.x - 50) * 0.1
    previous = (rope.angle + pi / 2) % (2 * pi) - pi
    rope.angle += rope.speed * 0.05
    current = (rope.angle + pi / 2) % (2 * pi) - pi
    reward = 0
    if previous * current < 0 and abs(current) < pi / 2 and self.mode == "jump":
      rope.count -= 1
      gap = 85 - rope.size - self.y
      reward = floor(100 / (gap if gap > 0 else 1))
      if not rope.count:
        self.rope = None
        self.mode = "stop"
    if self.mode == "stop":
      self.x += (5 - self.x) * 0.1
      if self.y == 84 and tap:
        self.mode = "run"
    if self.mode == "run":
      self.x += 2
      if self.x > 50:
        self.x, self.mode = 50, "stand"
    if self.mode == "stand" and self.held:
      self.vy, self.mode = -3, "jump"
    if self.mode in ("jump", "stop"):
      self.vy += 0.1 if self.held else 0.2
      self.y += self.vy
      if self.y > 84:
        self.y = 84
        if self.mode == "jump":
          self.mode = "stand"
    rx, ry = rope.position()
    self.terminal = abs(self.x - rx) < 3 + rope.size / 2 and abs(self.y - ry) < 6 + rope.size / 2
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), int(self.x // 10), 0] = True
    state[9, :, 1] = True
    count = 0
    if self.rope is not None:
      rope = self.rope
      count = rope.count
      x, y = rope.position()
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 2] = True
      future = rope.angle + rope.speed * 0.2
      x = rope.x + cos(future) * rope.radius
      y = 91 - rope.size / 2 - rope.radius + sin(future) * rope.radius
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 3] = True
      state[8, int(min(99, rope.x) // 10), 4] = True
    values = (self.steps, count, ("stop", "run", "stand", "jump").index(self.mode),
              self.held, min(511, round((self.vy + 4) * 16)))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
