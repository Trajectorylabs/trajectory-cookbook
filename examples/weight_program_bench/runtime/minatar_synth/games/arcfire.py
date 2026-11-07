from math import atan2, cos, hypot, pi, sin, sqrt

import numpy as np


HORIZON = 400
SECTORS = 24
MAX_CHARGE = 12
SPAWN_INTERVAL = 6
CENTER = (5, 5)
MAX_ENEMIES = (HORIZON + SPAWN_INTERVAL - 1) // SPAWN_INTERVAL


class ArcfireEnvironment:
  score_cap = MAX_ENEMIES * (MAX_ENEMIES + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = CENTER
    self.steps = self.last_action = self.aim = self.charge = self.arc_start = 0
    self.multiplier = 1
    self.spawn_index = int(self.random.integers(36))
    self.enemies: list[tuple[int, int]] = []
    self.shot: tuple[int, int, int, int] | None = None
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  @staticmethod
  def _range(width: int) -> int:
    return max(3, round(9 / sqrt(width)))

  def _in_arc(self, cell: tuple[int, int], start: int, width: int, radius: int) -> bool:
    dy, dx = cell[0] - self.player[0], cell[1] - self.player[1]
    direction = atan2(dy, dx) * SECTORS / (2 * pi)
    return (direction - start) % SECTORS <= width and abs(hypot(dx, dy) - radius) <= 0.75

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.steps += 1
    self.aim = (self.aim + 1) % SECTORS
    if action == 5 and self.last_action != 5:
      self.charge, self.arc_start, self.multiplier = 1, self.aim, 1
    elif action == 5 and self.charge:
      self.charge += 1
      if self.charge > MAX_CHARGE:
        self.charge = 0
    elif self.charge:
      width = self.charge
      reach = self._range(width)
      if self.shot is None:
        self.shot = (self.arc_start, width, reach, 1)
      angle = (self.arc_start + width / 2) * 2 * pi / SECTORS
      stride = reach / 2
      row, column = self.player
      self.player = (max(1, min(8, row + round(stride * sin(angle)))),
                     max(1, min(8, column + round(stride * cos(angle)))))
      self.charge = 0
    self.last_action = action

    if self.steps % SPAWN_INTERVAL == 1:
      if self.random.random() < 0.1:
        self.spawn_index = int(self.random.integers(36))
      else:
        self.spawn_index = (self.spawn_index + int(self.random.integers(-1, 2))) % 36
      side, offset = divmod(self.spawn_index, 9)
      self.enemies.append(((0, offset), (offset, 9), (9, 9 - offset), (9 - offset, 0))[side])
    if self.steps % 3 == 0:
      self.enemies = [(row + int(np.sign(CENTER[0] - row)), column + int(np.sign(CENTER[1] - column)))
                      for row, column in self.enemies]

    reward = 0
    if self.shot is not None:
      start, width, reach, radius = self.shot
      survivors = []
      for cell in self.enemies:
        if self._in_arc(cell, start, width, radius):
          reward += self.multiplier
          self.multiplier += 1
        else:
          survivors.append(cell)
      self.enemies = survivors
      self.shot = (start, width, reach, radius + 1) if radius < reach else None
    self.terminal = any(cell in (CENTER, self.player) for cell in self.enemies)
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for cell in self.enemies:
      state[*cell, 1] = True
    state[*CENTER, 2] = True
    for row in range(10):
      for column in range(10):
        cell = row, column
        if self.shot is not None:
          start, width, _, radius = self.shot
          state[row, column, 3] = self._in_arc(cell, start, width, radius)
        if self.charge:
          state[row, column, 4] = self._in_arc(cell, self.arc_start, self.charge, self._range(self.charge))
    values = (self.steps, self.aim, self.charge, self.arc_start, self.multiplier,
              self.spawn_index, int(self.last_action == 5), *(self.shot or (0, 0, 0, 0)))
    offset = 0
    for value, width in zip(values, (9, 5, 4, 5, 7, 6, 1, 5, 4, 4, 4), strict=True):
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    return state
