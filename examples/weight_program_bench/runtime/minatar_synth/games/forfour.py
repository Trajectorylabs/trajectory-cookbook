import math

import numpy as np


SIZE = 9
CENTER = SIZE // 2
CORE = (CENTER, CENTER)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class ForfourEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.board = np.full((SIZE, SIZE), -1, dtype=np.int8)
    self.board[CORE] = 0
    self.board[CENTER, CENTER + 1:CENTER + 4] = self.random.integers(1, 5, size=3)
    self._spawn()

  def _spawn(self) -> None:
    extent = int(np.max(np.abs(np.argwhere(self.board >= 0) - CENTER))) + 1
    self.color = int(self.random.integers(1, 5))
    if self.random.random() < math.sqrt(extent) * 0.1:
      self.color = 0
      extent = math.ceil(extent * 0.7)
    lane = int(self.random.integers(CENTER - extent + 1, CENTER + extent))
    self.direction = int(self.random.integers(4))
    self.row, self.column = ((lane, -1), (-1, lane), (lane, SIZE), (SIZE, lane))[self.direction]
    self.timer = 60

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
      reward += self._advance()
      if tap and frame == 0 and not self.terminal:
        self.board = np.rot90(self.board, -1).copy()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.timer -= 1
    if self.timer > 0:
      return 0
    self.timer = 6
    dy, dx = DIRECTIONS[self.direction]
    next_cell = self.row + dy, self.column + dx
    if not self._inside(*next_cell):
      self._spawn()
      return 0
    hit = int(self.board[next_cell])
    if hit < 0:
      self.row, self.column = next_cell
      return 0
    if not self._inside(self.row, self.column):
      self.terminal = True
      return 0
    color = self.color or hit or int(self.random.integers(1, 5))
    cell = self.row, self.column
    self.board[cell] = color
    group = self._connected(cell, color)
    reward = 0
    if len(group) >= 4:
      for removed in group:
        self.board[removed] = -1
      attached = self._connected(CORE)
      count = len(group)
      for row, column in np.argwhere(self.board > 0):
        if (row, column) not in attached:
          self.board[row, column] = -1
          count += 1
      reward = (count - 3)**2
    self._spawn()
    return reward

  @staticmethod
  def _inside(row: int, column: int) -> bool:
    return 0 <= row < SIZE and 0 <= column < SIZE

  def _connected(self, start: tuple[int, int], color: int | None = None) -> set[tuple[int, int]]:
    reached = {start}
    pending = [start]
    while pending:
      row, column = pending.pop()
      for dy, dx in DIRECTIONS:
        cell = row + dy, column + dx
        if self._inside(*cell) and cell not in reached:
          value = self.board[cell]
          if (value >= 0 if color is None else value == color):
            reached.add(cell)
            pending.append(cell)
    return reached

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for color in range(5):
      state[:SIZE, :SIZE, color] = self.board == color
    if self.direction % 2 == 0:
      state[self.row, :SIZE, 5] = True
    else:
      state[:SIZE, self.column, 5] = True
    row, column = min(8, max(0, self.row)), min(8, max(0, self.column))
    state[row, column, self.color] = True
    state[9, self.color, 5] = True
    state[9, 5 + self.direction, 5] = True
    state[9, 9, 5] = self.pressed
    for bit in range(6):
      state[bit, 9, 5] = bool(self.timer & (1 << bit))
    return state
