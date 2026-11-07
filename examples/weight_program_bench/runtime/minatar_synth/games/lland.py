from dataclasses import dataclass

import numpy as np


@dataclass
class Terrain:
  y: float
  pad: bool = False


class LlandEnvironment:
  score_cap = 10

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.terrain = [Terrain(49, True) if i == 4 else Terrain(90 - i) for i in range(9)]
    self.y, self.velocity = 30.0, 0.0
    self.offset = self.distance = self.index = self.steps = 0
    self.pad_countdown = 7
    self.landed = self.unlocked = self.pressed = self.terminal = False

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

  def _positions(self):
    for i, tile in enumerate(self.terrain):
      yield (i * 13 + self.offset) % 117 - 13, tile

  def _advance(self, tap: bool) -> int:
    if self.landed and not tap:
      return 0
    self.landed = False
    positions = list(self._positions())
    contacts = [tile for x, tile in positions if x < 28 and x + 13 > 22 and self.y + 3 > tile.y]
    missed_pad = any(tile.pad and x <= 0 < x + 13 for x, tile in positions)
    self.offset -= 1
    self.distance -= 1
    if self.distance < 0:
      pad = self.pad_countdown == 0
      low, high = (40, 70) if pad else (70 if self.pad_countdown > 7 or self.pad_countdown == 1 else 40, 90)
      self.terrain[self.index] = Terrain(self.random.uniform(low, high), pad)
      self.index = (self.index + 1) % 9
      self.distance += 13
      self.pad_countdown = 9 if pad else self.pad_countdown - 1
    if self.unlocked:
      self.velocity -= 0.4 * tap + 0.2 * self.pressed
    self.velocity = (self.velocity + 0.1) * 0.99
    if self.y < 0 and self.velocity < 0:
      self.velocity *= -1
    self.y += self.velocity
    self.terminal = missed_pad or any(not tile.pad for tile in contacts)
    for tile in contacts:
      if tile.pad:
        self.y, self.velocity = tile.y - 3, 0
        self.landed = self.unlocked = True
        for terrain in self.terrain:
          terrain.pad = False
        return 1
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), 2, 0] = True
    for x, tile in self._positions():
      left, right = max(0, int(x // 10)), min(9, int((x + 13) // 10))
      state[int(tile.y // 10):, left:right + 1, 2 if tile.pad else 1] = True
    future = min(9, max(0, int((self.y + 4 * self.velocity) // 10)))
    state[future, 2, 3] = True
    state[:, 2, 4] = True
    values = (self.steps, round((self.velocity + 12) * 16), round(self.y * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, :3, 5] = self.pressed, self.landed, self.unlocked
    return state
