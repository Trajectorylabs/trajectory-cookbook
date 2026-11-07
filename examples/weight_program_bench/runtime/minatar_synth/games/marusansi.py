import numpy as np


class MarusansiEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.grid = np.zeros((12, 6), dtype=np.int8)
    self.blocks = (1, 2)
    self.cursor = self.steps = self.action = self.timer = self.delay = 0
    self.multiplier = 1
    self.terminal = False
    self.marked = np.zeros_like(self.grid, dtype=bool)
    self.pending = None
    self._next_row()

  def _next_row(self):
    self.next_row = []
    for col in range(6):
      colors = [c for c in (1, 2, 3)
                if not (self.grid[0, col] == self.grid[1, col] == c)
                and self.next_row[-2:] != [c, c]]
      self.next_row.append(int(self.random.choice(colors)))

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(self.action if frame == 0 and self.action != previous else 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, command):
    if self.delay:
      self.delay -= 1
      if self.delay:
        return 0
      if self.pending is not None:
        changed = self.pending > 0
        self.grid[changed] = self.pending[changed]
        self.marked |= changed
        self.pending = None
        reward = self._match() if self.multiplier < 32 else 0
        if not self.delay:
          self.grid[self.marked] = 0
          self.delay = 5
        return reward
      for col in range(6):
        values = self.grid[:, col]
        remaining = values[values != 0]
        values[:] = 0
        values[:len(remaining)] = remaining
      self.marked[:] = False
      self.timer = 0
      reward = self._match()
      self._next_row()
      return reward
    heights = np.count_nonzero(self.grid, axis=0)
    if np.all(heights >= 11):
      self.terminal = True
      return 0
    self.timer += 1
    if self.timer > 400 or command == 4:
      self.grid[1:] = self.grid[:-1].copy()
      self.grid[0] = self.next_row
      self._next_row()
      self.timer = 0
    elif command in (1, 3):
      self.cursor = (self.cursor + (1 if command == 3 else -1)) % 6
    elif command == 2:
      self.blocks = self.blocks[::-1]
    elif command == 5 and heights[self.cursor] < 11:
      height = heights[self.cursor]
      self.grid[height:height + 2, self.cursor] = self.blocks[::-1]
      pair = tuple(int(c) for c in self.random.integers(1, 4, 2))
      self.blocks = ((pair[0] % 3 + 1, pair[1]) if pair == self.blocks else pair)
      self.multiplier = 1
      return self._match()
    return 0

  def _match(self):
    changes = np.zeros_like(self.grid)
    count = 0
    for board, output in ((self.grid.T, changes.T), (self.grid, changes)):
      for line, dest in zip(board, output, strict=True):
        start = 0
        while start < len(line):
          end = start + 1
          while end < len(line) and line[end] == line[start]:
            end += 1
          color = int(line[start])
          if color and end - start >= 3:
            others = [c for c in (1, 2, 3) if c != color]
            others.sort(key=lambda c: (c - color) % 3)
            for index in range(start, end):
              dest[index] = others[(index - start) % 2]
            count += end - start
          start = end
    if not count:
      return 0
    self.pending = changes
    self.delay = 50
    reward = count * self.multiplier
    self.multiplier += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for row, col in np.argwhere(self.grid > 0):
      state[row % 8, col, 3 * (row // 8) + self.grid[row, col] - 1] = True
    for col, color in enumerate(self.next_row):
      state[8, col, color - 1] = True
    for col, color in enumerate(self.blocks):
      state[9, col, color - 1] = True
    state[9, self.cursor, 3] = True
    for col, value in enumerate((self.timer, self.delay, self.multiplier, self.action), 6):
      for bit in range(9):
        state[bit, col, 0] = bool(value & (1 << bit))
    return state
