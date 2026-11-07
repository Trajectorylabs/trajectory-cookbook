from dataclasses import dataclass

import numpy as np


HORIZON = 400
SPAWN_INTERVAL = 8
MAX_FISH = (HORIZON // SPAWN_INTERVAL) * 13
BIRD_ROW = 2
NORMAL, BIG, FAKE, EYE = range(4)
FLIGHT = (8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9)
FRAGMENTS = ((0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3),
             (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 5))


@dataclass
class Fish:
  column: int
  kind: int = NORMAL
  age: int = 0
  row_offset: int = 0

  @property
  def row(self) -> int:
    return max(0, min(9, FLIGHT[self.age] + self.row_offset))


class BSFishEnvironment:
  score_cap = MAX_FISH * (1 + 5 * (MAX_FISH - 1))

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.speed = 0
    self.multiplier = 1
    self.special_in = 3
    self.fish: list[Fish] = []
    self.terminal = False

  @property
  def bird_column(self) -> int:
    return 1 + self.speed // 2

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
    self.speed = min(6, self.speed + 1) if action == 5 else max(0, self.speed - 1)
    if self.steps % SPAWN_INTERVAL == 1:
      if self.special_in == 0:
        kind = int(self.random.choice((BIG, FAKE)))
        self.special_in = int(self.random.integers(3, 9))
      else:
        kind = NORMAL
        self.special_in -= 1
      self.fish.append(Fish(int(self.random.integers(7, 10)), kind))
    reward = 0
    remaining = []
    for fish in self.fish:
      fish.age += 1
      if self.speed >= 4 or self.steps % 3 == 0:
        fish.column -= 1
      if fish.age == len(FLIGHT) or fish.column < -1:
        if fish.kind != BIG:
          self.multiplier = max(1, self.multiplier - 1)
        continue
      if fish.kind == FAKE and fish.row <= 5:
        remaining.extend(Fish(fish.column + column - 2, EYE if (row, column) == (1, 2) else NORMAL,
                              fish.age, row - 1) for row, column in FRAGMENTS)
      elif fish.kind == BIG:
        if abs(fish.row - BIRD_ROW) <= 1 and abs(fish.column - self.bird_column) <= 1:
          self.terminal = True
        remaining.append(fish)
      elif fish.row == BIRD_ROW and fish.column == self.bird_column:
        reward += self.multiplier
        self.multiplier += 10 if fish.kind == EYE else 1
      else:
        remaining.append(fish)
    self.fish = remaining
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[BIRD_ROW, self.bird_column, 0] = True
    for fish in self.fish:
      if 0 <= fish.column < 10:
        channel = (2, 1, 1, 3)[fish.kind]
        state[fish.row, fish.column, channel] = True
        next_row = FLIGHT[min(fish.age + 1, len(FLIGHT) - 1)] + fish.row_offset
        if 0 <= next_row < 10:
          state[next_row, fish.column, 4] = True
    for row, value in enumerate((self.steps, self.speed, self.multiplier, self.special_in, self.last_action)):
      for bit in range(13):
        index = row * 13 + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
    return state
