from collections import deque
from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Segment:
  x: float
  y: float
  dx: float
  dy: float
  born: int

  def overlaps(self, x: float, y: float) -> bool:
    if abs(x - self.x) > abs(self.dx) + 3 or abs(y - self.y) > abs(self.dy) + 3:
      return False
    length_squared = self.dx**2 + self.dy**2
    fraction = ((x - self.x) * self.dx + (y - self.y) * self.dy) / length_squared
    fraction = min(1, max(0, fraction))
    return math.hypot(x - self.x - fraction * self.dx, y - self.y - fraction * self.dy) < 3


class ChalkboardEnvironment:
  score_cap = HORIZON * FRAMES_PER_STEP * 3 // 10

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = self.heading = self.streak = 0
    self.turn_timer = 120
    self.pressed = self.terminal = False
    self.x = self.y = self.score_carry = 0.0
    self.write_x = self.write_y = 0.0
    self.energy = 60.0
    self.segments = deque(
      (Segment(70, y, 0, 10, -4) for y in range(-40, 40, 10)),
      maxlen=1000,
    )

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    changed = pressed != self.pressed
    self.pressed = pressed
    if changed:
      self.streak = 0
    self.steps += 1
    reward = 0
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(changed and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, changed: bool) -> int:
    self.frame += 1
    self.turn_timer -= 1
    if not self.turn_timer:
      self.heading = (self.heading + 1) % 4
      self.turn_timer = int(self.random.integers(30, 150))
    dx, dy = HEADINGS[self.heading]
    self.x += 0.9 * dx
    self.y += 0.9 * dy
    self.streak = min(120, self.streak + 1)
    amount = 0.3 * (self.streak / 120)**1.5
    reward = 0
    if self.pressed:
      if changed:
        self.write_x, self.write_y = self.x, self.y
      if any(self.frame - line.born >= 4 and line.overlaps(self.x, self.y) for line in self.segments):
        self.energy -= 10
      else:
        self.energy += amount
        self.score_carry += amount
        reward = int(self.score_carry)
        self.score_carry -= reward
      dx, dy = self.x - self.write_x, self.y - self.write_y
      if dx * dx + dy * dy >= 4:
        self.segments.append(Segment(self.write_x, self.write_y, dx, dy, self.frame))
        self.write_x, self.write_y = self.x, self.y
    else:
      self.energy -= amount
    self.energy = min(100, max(0, self.energy))
    self.terminal = self.energy <= 0
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    for line in self.segments:
      x, y = line.x - self.x + 50, line.y - self.y + 50
      if not (-12 <= x < 112 and -12 <= y < 112):
        continue
      channel = 2 if self.frame - line.born < 4 else 1
      for fraction in (0, 0.5, 1):
        px, py = x + line.dx * fraction, y + line.dy * fraction
        if 0 <= px < 100 and 0 <= py < 100:
          state[int(py) // 10, int(px) // 10, channel] = True
    dx, dy = HEADINGS[self.heading]
    state[5 + dy, 5 + dx, 3] = True
    if self.turn_timer <= 30:
      dx, dy = HEADINGS[(self.heading + 1) % 4]
      state[5 + dy, 5 + dx, 4] = True
    for row, value in enumerate((self.steps, round(self.energy), self.turn_timer, self.streak)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
