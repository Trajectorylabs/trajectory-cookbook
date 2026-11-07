import numpy as np


SIZE = 9
CENTER = SIZE // 2
EMPTY = -1
HORIZON = 400
GROWTH_INTERVAL = 30
SETTLE_DELAY = 15
DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))


class BreedCEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.growth_count = self.settle = self.clears = 0
    self.multiplier = 1
    self.timer = GROWTH_INTERVAL
    self.terminal = False
    self.cursor = CENTER, CENTER
    self.grid = np.full((SIZE, SIZE), EMPTY, dtype=np.int8)
    self.grid[CENTER, CENTER] = self.random.integers(4)
    self._grow()

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    select = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    reward = 0
    if 1 <= action <= 4:
      dy, dx = DIRECTIONS[action - 1]
      y, x = self.cursor
      self.cursor = min(SIZE - 1, max(0, y + dy)), min(SIZE - 1, max(0, x + dx))
    if self.settle:
      self.settle -= 1
      if not self.settle:
        self._compact()
        self.clears += 1
        self.timer = GROWTH_INTERVAL
    else:
      if self.grid[CENTER, CENTER] == EMPTY:
        self.grid[CENTER, CENTER] = self.random.integers(4)
      if select:
        reward = self._erase()
      if not self.settle:
        self.timer -= 1
        if self.timer <= 0:
          self._grow()
    self.terminal = self.steps >= self.horizon or bool(np.all(self.grid != EMPTY))
    return float(reward), self.terminal

  def _erase(self) -> int:
    color = self.grid[self.cursor]
    group = set()
    if color != EMPTY:
      group.add(self.cursor)
      pending = [self.cursor]
      while pending:
        y, x = pending.pop()
        for dy, dx in DIRECTIONS:
          ny, nx = y + dy, x + dx
          neighbor = ny, nx
          if 0 <= ny < SIZE and 0 <= nx < SIZE and neighbor not in group and self.grid[neighbor] == color:
            group.add(neighbor)
            pending.append(neighbor)
    size = len(group)
    if size < 4:
      self._grow()
      self._grow()
      return -size
    for cell in group:
      self.grid[cell] = EMPTY
    reward = size * self.multiplier
    self.multiplier += 1
    self.settle = SETTLE_DELAY
    return reward

  def _grow(self) -> None:
    lines = self.grid if self.growth_count % 2 == 0 else self.grid.T
    old = lines.copy()
    lines[:, :CENTER - 1] = old[:, 1:CENTER]
    lines[:, CENTER + 2:] = old[:, CENTER + 1:-1]
    for line in lines:
      color = int(line[CENTER])
      if color != EMPTY:
        direction = int(self.random.choice((-1, 1)))
        new_color = (color + int(self.random.integers(3))) % 4
        line[CENTER - direction] = color
        line[CENTER] = line[CENTER + direction] = new_color
    self.growth_count += 1
    self.multiplier = 1
    self.timer = GROWTH_INTERVAL

  def _compact(self) -> None:
    axes = (self.grid, self.grid.T)
    if self.clears % 2:
      axes = axes[::-1]
    moved = True
    while moved:
      moved = False
      for lines in axes:
        for line in lines:
          for start, stop, step in ((CENTER, 0, -1), (CENTER, SIZE - 1, 1)):
            for index in range(start, stop, step):
              source = index + step
              if line[index] == EMPTY and line[source] != EMPTY:
                line[index], line[source] = line[source], EMPTY
                moved = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for color in range(4):
      state[:SIZE, :SIZE, color] = self.grid == color
    state[*self.cursor, 4] = True
    for row, value in enumerate((self.steps, self.timer, self.settle, self.multiplier)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.last_action == 5
    state[9, 1, 5] = self.growth_count % 2 == 1
    return state
