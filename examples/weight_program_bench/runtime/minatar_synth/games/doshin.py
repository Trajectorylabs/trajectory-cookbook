from dataclasses import dataclass

import numpy as np


CENTERS = (25, 75, 125, 175)


@dataclass
class Walker:
  x: float
  speed: float
  friendly: bool
  y: float = 87
  vy: float = 0
  struck: bool = False


class DoshinEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.until_friend = 0
    self.terminal = False
    self.cooldowns = [0] * len(CENTERS)
    self.walkers: list[Walker] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    press = action - 1 if 1 <= action <= 4 and action != self.last_action else None
    self.last_action = action
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance_frame(press if frame == 0 else None)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, press: int | None) -> int:
    if self.random.random() < 0.03:
      self.until_friend -= 1
      friendly = self.until_friend < 0
      if friendly:
        speed = 0.6 + 0.2 * int(self.random.integers(2))
        self.until_friend = int(self.random.integers(10, 20))
      else:
        speed = 0.1 + 0.2 * int(self.random.integers(4))
      self.walkers.append(Walker(0, speed, friendly))
    active = press is not None and self.cooldowns[press] == 0
    self.cooldowns = [max(0, timer - 1) for timer in self.cooldowns]
    if active:
      self.cooldowns[press] = 60

    survivors = []
    reward, combo = 0, 1
    for walker in self.walkers:
      walker.x += walker.speed
      walker.y += walker.vy
      if walker.struck:
        walker.vy += 0.1
        if walker.y > 99:
          if walker.friendly:
            self.terminal = True
          else:
            reward += combo
            combo += 1
          continue
      else:
        if active and abs(walker.x - CENTERS[press]) < 18:
          walker.struck = True
          walker.vy = -1
        if walker.x > 199:
          if walker.friendly:
            reward += 10
          else:
            self.terminal = True
          continue
      survivors.append(walker)
    self.walkers = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for index, (center, timer) in enumerate(zip(CENTERS, self.cooldowns)):
      column = center // 20
      row = min(9, (30 + timer) // 10)
      state[:row, column, 0] = True
      state[0, column, 1] = timer == 0
      for bit in range(6):
        state[index, bit, 5] = bool(timer & (1 << bit))
    for walker in self.walkers:
      if 0 <= walker.x < 200 and 0 <= walker.y < 100:
        row, col = int(walker.y) // 10, int(walker.x) // 20
        state[row, col, 2 if walker.friendly else 3] = True
        state[row, col, 4] |= walker.struck
    for bit in range(9):
      state[8, bit, 5] = bool(self.steps & (1 << bit))
    state[9, self.last_action, 5] = True
    return state
