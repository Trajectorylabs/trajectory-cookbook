from dataclasses import dataclass

import numpy as np


HORIZON = 300
RED_INTERVAL = 12
ORDINARY_INTERVAL = 4
MAX_RED = (HORIZON + RED_INTERVAL - 1) // RED_INTERVAL
MAX_BALLS = HORIZON // ORDINARY_INTERVAL + 10 * MAX_RED
MOVES = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


@dataclass
class Ball:
  row: int
  column: int
  red: bool = False


class BBlastEnvironment:
  score_cap = MAX_BALLS * (MAX_BALLS + 1) // 2 + 10 * MAX_RED

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.cursor = (5, 5)
    self.center = self.cursor
    self.steps = self.last_action = self.charge = self.cooldown = 0
    self.balls: list[Ball] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.steps += 1
    self.cooldown = max(0, self.cooldown - 1)
    reward = 0
    if action == 5:
      if not self.charge:
        self.center = self.cursor
        self.charge = 1
      elif not self.cooldown or self.steps % 3 == 0:
        self.charge = min(12, self.charge + 1)
    else:
      if self.charge:
        reward += self._blast()
      if action in MOVES:
        dr, dc = MOVES[action]
        row, column = self.cursor
        self.cursor = max(0, min(9, row + dr)), max(0, min(9, column + dc))
    self.last_action = action
    if self.steps % RED_INTERVAL == 1:
      self.balls.append(Ball(1, int(self.random.integers(2, 8)), True))
    if self.steps % ORDINARY_INTERVAL == 0:
      self.balls.append(Ball(int(self.random.integers(2, 8)), int(self.random.integers(1, 9))))
    if self.steps % 2 == 0:
      reward += self._settle()
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _blast(self) -> int:
    row, column = self.center
    hit = [ball for ball in self.balls
           if (ball.row - row) ** 2 + (ball.column - column) ** 2 <= (self.charge / 2 + 0.5) ** 2]
    removed = {id(ball) for ball in hit}
    self.balls = [ball for ball in self.balls if id(ball) not in removed]
    for ball in hit:
      if ball.red:
        self.balls.extend(Ball(max(0, min(9, ball.row + dr)), max(0, min(9, ball.column + dc)))
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    self.charge = 0
    self.cooldown = 6
    return len(hit) * (len(hit) + 1) // 2

  def _settle(self) -> int:
    reward = 0
    occupied: set[tuple[int, int]] = set()
    for ball in sorted(self.balls, key=lambda item: -item.row):
      options = [(ball.row + 1, ball.column), (ball.row, ball.column),
                 (ball.row, ball.column - 1), (ball.row, ball.column + 1),
                 (ball.row - 1, ball.column)]
      for row, column in options:
        if 0 <= row < 10 and 0 <= column < 10 and (row, column) not in occupied:
          ball.row, ball.column = row, column
          occupied.add((row, column))
          break
      if ball.red and ball.row == 9:
        reward += 10
        ball.red = False
      if ball.row == 0:
        self.terminal = True
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.cursor, 0] = True
    for ball in self.balls:
      state[ball.row, ball.column, 1 if ball.red else 2] = True
    if self.charge:
      state[*self.center, 3] = True
      rows, columns = np.ogrid[:10, :10]
      state[:, :, 4] = ((rows - self.center[0]) ** 2 + (columns - self.center[1]) ** 2
                       <= (self.charge / 2 + 0.5) ** 2)
    for row, value in enumerate((self.steps, self.charge, self.cooldown, self.last_action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
