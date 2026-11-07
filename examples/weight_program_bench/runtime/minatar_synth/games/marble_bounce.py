from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Pin:
  x: float
  y: float
  size: float
  last_hit: int = 0


class MarbleBounceEnvironment:
  score_cap = 1500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 80.0
    self.vx = self.vy = 0.0
    self.steps = self.frames = self.timer = self.target_since = 0
    self.interval = 360
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.pins: list[Pin] = []
    self._wave(2)

  def _wave(self, size):
    for _ in range(8):
      pin = self._sample_pin(50, size)
      if pin is None:
        pin = Pin(self.random.uniform(10, 90), self.random.uniform(20, 80), size)
      self.pins.append(pin)
    self.target = self.pins[0]

  def _sample_pin(self, attempts, size):
    for _ in range(attempts):
      x, y = self.random.uniform(10, 90), self.random.uniform(20, 80)
      if math.hypot(x - self.x, y - self.y) >= 15 and all(
          math.hypot(x - pin.x, y - pin.y) >= 15 for pin in self.pins):
        return Pin(x, y, size)
    return None

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
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

  def _distance(self, pin):
    return math.hypot(self.x - pin.x, self.y - pin.y)

  def _advance(self, tap):
    self.vy += 0.15
    if tap:
      self.vy = min(5, max(-5, self.vy - 3))
    self.timer += 1
    if self.timer >= self.interval:
      self.timer = 0
      pin = self._sample_pin(20, 0)
      if pin is not None:
        self.pins.append(pin)
        self.interval = max(90, self.interval - 30)
    self.x += self.vx
    self.y += self.vy
    if self.y >= 89.5:
      self.y = 89.5
      self.vy *= -0.7
      if abs(self.vx) < 0.5:
        self.vx += self.random.uniform(1, 2) * self.random.choice((-1, 1))
    if self.frames - self.target_since > 900:
      self.terminal = True
      return 0
    for pin in self.pins:
      pin.size = min(2, pin.size + 0.1)
    reward = self._hit_pins()
    if not 5 <= self.x <= 95:
      self.vx = abs(self.vx) if self.x < 5 else -abs(self.vx)
      self.x = min(95, max(5, self.x))
    if self.y < 5:
      self.y = 5
      self.vy = abs(self.vy)
    self.frames += 1
    return reward

  def _hit_pins(self):
    reward = 0
    regular = min((pin for pin in self.pins if pin is not self.target),
                  key=self._distance, default=None)
    if regular is not None and self._distance(regular) < 3 + regular.size:
      angle = math.atan2(self.y - regular.y, self.x - regular.x)
      separation = 3 + regular.size
      self.x = regular.x + math.cos(angle) * separation
      self.y = regular.y + math.sin(angle) * separation
      if self.frames - regular.last_hit > 9:
        reward += self.multiplier
        regular.last_hit = self.frames
      self.vx = self.random.uniform(1, 2) * self.random.choice((-1, 1))
      self.vy = -abs(self.vy) * 0.8
    if self._distance(self.target) < 3 + self.target.size:
      angle = math.atan2(self.y - self.target.y, self.x - self.target.x)
      self.x += 2 * math.cos(angle)
      self.y += 2 * math.sin(angle)
      self.vx = self.random.uniform(2, 3) * self.random.choice((-1, 1))
      self.vy = -abs(self.vy) * 0.9
      reward += 100 * self.multiplier
      self.pins = [pin for pin in self.pins if pin is not self.target]
      if self.pins:
        self.target = self.pins[self.random.integers(len(self.pins))]
      else:
        self.multiplier += 1
        self._wave(0)
      self.interval, self.timer, self.target_since = 360, 0, self.frames
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), min(9, max(0, int(self.x // 10))), 0] = True
    for pin in self.pins:
      state[int(pin.y // 10), int(pin.x // 10), 2 if pin is self.target else 1] = True
    state[9, :, 3] = True
    if self.frames - self.target_since > 600:
      state[int(self.target.y // 10), int(self.target.x // 10), 4] = True
    gauges = (self.steps, (self.frames - self.target_since) // 2, self.multiplier,
              round((self.vx + 8) * 20), round((self.vy + 8) * 20))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
