from math import ceil, cos, hypot, pi, sin

import numpy as np


HORIZON = 400
UNIT = 8
EDGE = 10 * UNIT
REST_LENGTH = 6
ANGLE_COUNT = 32
PIN_RADIUS = 2


class PinClimbEnvironment:
  score_cap = HORIZON * ceil(hypot(EDGE - 1, EDGE - 1))

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.angle = self.last_action = self.scroll = 0
    self.length = REST_LENGTH
    self.anchor = (40, 4)
    self.pins = [self.anchor]
    self.next_pin = 4
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _catch(self) -> float:
    angle = self.angle * 2 * pi / ANGLE_COUNT
    horizontal, vertical = cos(angle), sin(angle)
    x, y = self.anchor
    target = None
    for pin in self.pins:
      if pin == self.anchor:
        continue
      dx, dy = pin[0] - x, pin[1] - y
      along = dx * horizontal + dy * vertical
      across = dx * vertical - dy * horizontal
      if 0 < along <= self.length and abs(across) <= PIN_RADIUS:
        target = pin
    if target is None:
      return 0.0
    self.anchor = target
    self.length = REST_LENGTH
    return float(ceil(hypot(target[0] - x, target[1] - y)))

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    self.angle = (self.angle + (2 if self.steps >= 200 else 1)) % ANGLE_COUNT
    if action == 5:
      self.length = min(EDGE * 2, self.length + 2)
    else:
      self.length -= (self.length - REST_LENGTH + 3) // 4
    advance = max(1, (64 - self.anchor[1]) // 8)
    self.scroll += advance
    x, y = self.anchor
    self.anchor = x, y + advance
    self.pins = [(x, y + advance) for x, y in self.pins if y + advance < EDGE]
    while self.next_pin <= self.scroll:
      pin = int(self.random.integers(UNIT, EDGE - UNIT)), self.scroll - self.next_pin
      self.pins.append(pin)
      self.next_pin += int(self.random.integers(4, 13))
    if self.anchor[1] >= EDGE:
      self.terminal = True
      return 0.0, True
    reward = self._catch()
    self.terminal = self.steps >= self.horizon
    return reward, self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = self.anchor
    state[min(9, y // UNIT), x // UNIT, 0] = True
    for px, py in self.pins:
      state[py // UNIT, px // UNIT, 2] = True
    angle = self.angle * 2 * pi / ANGLE_COUNT
    for distance in range(self.length + 1):
      px = round(x + cos(angle) * distance)
      py = round(y + sin(angle) * distance)
      if 0 <= px < EDGE and 0 <= py < EDGE:
        state[py // UNIT, px // UNIT, 3] = True
    values = (self.steps, self.angle, self.length, self.next_pin - self.scroll,
              int(self.last_action == 5), x, y)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
