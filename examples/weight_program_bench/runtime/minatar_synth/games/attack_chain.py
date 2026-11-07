from math import sqrt

import numpy as np


HORIZON = 400
CARDS = 6
CHAINS_PER_DEAL = 7
RED, GREEN, BLUE = range(3)
PRESSURE_LIMIT = 300


class AttackChainEnvironment:
  # One chain per two steps at most, and no card reaches 200 damage.
  score_cap = (HORIZON // 2) * (99 + 250) * (1 + HORIZON // 2)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.pressure = self.total = self.turn = self.chains = 0
    self.cooldown = self.pending = 0
    self.last_type = -1
    self.last_action = 0
    self.multiplier = 1
    self.terminal = False
    self._deal()

  def _deal(self) -> None:
    green_left = 2
    self.types = []
    for _ in range(CARDS):
      if green_left and self.random.random() < 0.2:
        self.types.append(GREEN)
        green_left -= 1
      else:
        self.types.append(int(self.random.choice((RED, BLUE))))
    scale = 4 + sqrt(min(20, self.multiplier))
    self.values = [int(value * scale) for value in self.random.integers(3, 7, size=CARDS)]
    self.available = [True] * CARDS

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    self.pressure += 1
    reward = 0
    if self.cooldown:
      self.cooldown = 0
      self.total += self.pending
      self.pending = 0
      if self.total >= 100 or not any(self.available):
        reward = self._settle()
    elif self.pressure >= PRESSURE_LIMIT:
      self.terminal = True
    elif self.available[action]:
      self.pressure += 4
      self.available[action] = False
      value, color = self.values[action], self.types[action]
      if color == RED and self.turn == 0:
        value = value * 5 // 4
      elif color == GREEN:
        value = min(value, 99 - self.total)
      self.pending = value
      self.cooldown = 1
      self.last_type = color
      self.turn += 1
      self.values[action] = self.values[action] * 11 // 10 + 10
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _settle(self) -> int:
    reward = self.total * self.multiplier
    excess = max(0, self.total - 115)
    self.pressure = max(0, self.pressure - int(excess * sqrt(excess / self.multiplier)))
    returns = sum(self.total >= threshold for threshold in (100, 150, 200))
    unavailable = [index for index, ready in enumerate(self.available) if not ready]
    if returns and unavailable and self.last_type == BLUE:
      best = max(unavailable, key=self.values.__getitem__)
      self.available[best] = True
      unavailable.remove(best)
      returns -= 1
    for index in self.random.permutation(unavailable)[:returns]:
      self.available[index] = True
    self.chains += 1
    if not any(self.available) or self.chains == CHAINS_PER_DEAL:
      self.chains = 0
      self.multiplier += 1
      self._deal()
    self.total = self.turn = 0
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for index, (color, value, ready) in enumerate(zip(self.types, self.values, self.available, strict=True)):
      state[9, index, color] = True
      state[9, index, 3] = ready
      for bit in range(8):
        state[bit, index, 4] = bool(value & (1 << bit))
    values = (self.pressure, self.total, self.turn, self.chains, self.multiplier,
              self.pending, self.cooldown, self.last_type + 1, self.steps, self.last_action)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
