from math import ceil, cos, sin

import numpy as np


HORIZON = 400
SPAWN_INTERVAL = 6
FLOOR = 9
MIN_LENGTH = 20
MAX_LENGTH = 64


class BalanceEnvironment:
  score_cap = ((HORIZON + SPAWN_INTERVAL - 1) // SPAWN_INTERVAL) * 3 * MAX_LENGTH

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.base = self.cursor = 4.5
    self.velocity = self.angle = self.angular_velocity = self.wind = 0.0
    self.length = MIN_LENGTH
    self.coin_column = float(self.random.choice((2, 7)))
    self.coins: list[tuple[float, float, float]] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  @property
  def tip(self) -> tuple[float, float]:
    return FLOOR - cos(self.angle) * self.length / 8, self.base + sin(self.angle) * self.length / 8

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    self.cursor = max(0.0, min(9.0, self.cursor + (1 if action == 3 else -1 if action == 1 else 0)))
    self.wind = 0.94 * (self.wind + float(self.random.uniform(-0.025, 0.025)))
    self.velocity = 0.90 * (self.velocity + 0.08 * (self.cursor - self.base) - self.wind)
    self.base += self.velocity
    self.angular_velocity = 0.88 * (self.angular_velocity - 0.16 * self.velocity
                                    + self.angle * self.length / 800)
    self.angle += self.angular_velocity
    row, column = self.tip
    shift = (4.5 - column) * 0.18
    self.base += shift
    self.cursor = max(0.0, min(9.0, self.cursor + shift))
    if row >= FLOOR - 0.25:
      self.length /= 2
      self.angular_velocity *= -0.5
      if self.length < MIN_LENGTH:
        self.terminal = True
        return 0.0, True
      self.angle /= 2
    if self.steps % SPAWN_INTERVAL == 1:
      self.coins.append((0.0, self.coin_column, float(self.random.uniform(-0.15, 0.15))))
      self.coin_column = (self.coin_column + shift + float(self.random.uniform(-0.6, 0.6))) % 10

    row, column = self.tip
    reward = 0
    remaining = []
    for y, x, drift in self.coins:
      y += 0.5
      drift = 0.96 * (drift + self.wind / 2)
      x += drift + shift
      tip_hit = abs(y - row) <= 0.45 and abs(x - column) <= 0.55
      fraction = (FLOOR - y) / max(0.01, FLOOR - row)
      shaft_x = self.base + (column - self.base) * fraction
      if tip_hit or (0 <= fraction <= 1 and abs(x - shaft_x) <= 0.45):
        for _ in range(3 if tip_hit else 1):
          reward += ceil(self.length)
          self.length = min(MAX_LENGTH, self.length + 1)
      elif y < FLOOR:
        remaining.append((y, x, drift))
    self.coins = remaining
    self.terminal = self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, column = self.tip
    state[min(9, max(0, round(row))), min(9, max(0, round(column))), 0] = True
    state[FLOOR, :, 1] = True
    for y, x, _ in self.coins:
      if 0 <= round(x) < 10:
        state[round(y), round(x), 2] = True
    for fraction in np.linspace(0, 1, 16):
      x = round(self.base + fraction * (column - self.base))
      y = round(FLOOR + fraction * (row - FLOOR))
      if 0 <= x < 10 and 0 <= y < 10:
        state[y, x, 3] = True
    state[FLOOR, round(self.cursor), 4] = True
    values = (self.steps, round(self.length * 4), round(self.angle * 64) + 256,
              round(self.angular_velocity * 128) + 256, round(self.velocity * 32) + 256,
              round(self.wind * 512) + 256, self.last_action)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
