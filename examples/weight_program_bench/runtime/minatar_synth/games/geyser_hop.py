from dataclasses import dataclass

import numpy as np


@dataclass
class Geyser:
  x: float
  height: int
  stomped: bool = False


class GeyserHopEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 40.0
    self.vy = 0.0
    self.ceiling = 3.0
    self.steps = self.frames = self.since_stomp = 0
    self.combo = 1
    self.pressed = self.terminal = False
    self.geysers = [Geyser(95, 40)]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.frames += 1
    self.since_stomp += 1
    if self.since_stomp > 200:
      self.ceiling += 0.1
    if self.frames % 91 == 61:
      self.geysers.append(Geyser(110, int(self.random.integers(40, 60))))
    self.x += (45 - self.x) * 0.01
    self.vy = min(3, max(-2.2, self.vy + (-0.16 if self.pressed else 0.13)))
    self.y += self.vy
    if self.y < self.ceiling or self.y > 92:
      self.terminal = True
      return 0
    reward = 0
    for geyser in self.geysers:
      geyser.x -= 1
      if geyser.x < -15 and not geyser.stomped:
        self.combo = max(1, self.combo - 1)
      if geyser.stomped:
        continue
      top = 92 - geyser.height
      dx = abs(self.x - geyser.x)
      if dx < 8 and self.y + 2 > top and self.y - 2 < top + 6:
        if self.vy <= 0:
          self.terminal = True
          break
        reward += self.combo
        self.combo = min(16, self.combo + 1)
        self.vy = -2.8
        self.y = top - 2
        geyser.height = 8
        geyser.stomped = True
        self.ceiling = max(3, self.ceiling - 8)
        self.since_stomp = 0
      elif dx < 7 and self.y + 2 > top + 4:
        self.terminal = True
        break
    self.geysers = [geyser for geyser in self.geysers if geyser.x >= -15]
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), int(self.x) // 10, 0] = True
    for geyser in self.geysers:
      top = (92 - geyser.height) // 10
      for col in range(max(0, int(geyser.x - 5) // 10), min(10, int(geyser.x + 5) // 10 + 1)):
        if not geyser.stomped:
          state[top, col, 1] = True
        state[top + int(not geyser.stomped):10, col, 3 if geyser.stomped else 2] = True
    state[:min(10, int(self.ceiling) // 10 + 1), :, 4] = True
    for row, value in enumerate((self.steps, self.combo, round((self.vy + 3) * 50), min(511, self.since_stomp))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
