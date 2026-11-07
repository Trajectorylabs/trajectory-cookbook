import numpy as np


HORIZON = 400
UNIT = 4
BIRD_COLUMN = 4
MAX_CHICKS = BIRD_COLUMN
MIN_PLATFORM_SPACING = 4


class InTowEnvironment:
  score_cap = MAX_CHICKS * (2 + HORIZON // MIN_PLATFORM_SPACING)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.y = 3 * UNIT
    self.velocity = 0
    self.steps = self.distance = self.last_action = self.chicks = 0
    self.grounded = False
    self.terminal = False
    self.platforms = [(7, 1, 7), (5, 8, 11)]
    self.pickups = {(4, 9)}
    self.bullets: set[tuple[int, int]] = set()
    self.history = [self.y] * MAX_CHICKS
    self.next_platform = 4
    self.next_bullet = 12

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _scroll(self) -> None:
    self.distance += 1
    self.platforms = [(row, left - 1, right - 1)
                      for row, left, right in self.platforms if right > 0]
    self.pickups = {(row, column - 1) for row, column in self.pickups if column > 0}
    if self.distance == self.next_platform:
      row = int(self.random.integers(3, 9))
      width = int(self.random.integers(3, 5))
      self.platforms.append((row, 9, 8 + width))
      self.pickups.add((row - 1, 9 + width // 2))
      self.next_platform += width + int(self.random.integers(1, 4))

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    self.history = [self.y, *self.history[:-1]]
    if self.steps % 2 == 0 or self.steps >= 200:
      self._scroll()
    if self.grounded:
      self.grounded = any(left <= BIRD_COLUMN <= right and (row - 1) * UNIT == self.y
                          for row, left, right in self.platforms)
    if pressed and (self.grounded or self.chicks):
      if not self.grounded:
        self.chicks -= 1
      self.grounded = False
      self.velocity = -5
    previous_y = self.y
    if not self.grounded:
      gravity = int(action != 5 or self.steps % 2 == 0)
      self.velocity = min(7, self.velocity + gravity)
      self.y = max(0, self.y + self.velocity)
      if self.velocity > 0:
        landings = [(row - 1) * UNIT for row, left, right in self.platforms
                    if left <= BIRD_COLUMN <= right and previous_y <= (row - 1) * UNIT <= self.y]
        if landings:
          self.y = min(landings)
          self.velocity = 0
          self.grounded = True
    self.bullets = {(row, column - 1) for row, column in self.bullets if column > 0}
    if self.steps == self.next_bullet:
      self.bullets.add((int(self.random.integers(1, 9)), 9))
      self.next_bullet += int(self.random.integers(6, 11))
    reward = 0
    for row, column in sorted(self.pickups):
      if column == BIRD_COLUMN and min(previous_y, self.y) // UNIT <= row <= max(previous_y, self.y) // UNIT:
        self.pickups.remove((row, column))
        self.chicks = min(MAX_CHICKS, self.chicks + 1)
        reward += self.chicks
    head = (self.y // UNIT, BIRD_COLUMN)
    if head in self.bullets:
      self.bullets.remove(head)
      if self.chicks:
        self.chicks = 0
        self.grounded = False
        self.velocity = 7
      else:
        self.terminal = True
    for index in range(self.chicks):
      tail = (self.history[index] // UNIT, BIRD_COLUMN - index - 1)
      if tail in self.bullets:
        self.chicks = index
        break
    self.terminal |= self.y >= 10 * UNIT or self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, self.y // UNIT), BIRD_COLUMN, 0] = True
    for row, column in self.bullets:
      state[row, column, 1] = True
    for row, column in self.pickups:
      if column < 10:
        state[row, column, 2] = True
    for row, left, right in self.platforms:
      state[row, max(0, left): min(10, right + 1)] = True
    for index in range(self.chicks):
      state[self.history[index] // UNIT, BIRD_COLUMN - index - 1, 4] = True
    values = (self.y, self.velocity + 5, self.steps, self.chicks, int(self.grounded),
              int(self.last_action == 5), self.next_platform - self.distance,
              self.next_bullet - self.steps, *self.history)
    for index, value in enumerate(values):
      for bit in range(7 if index != 2 else 9):
        offset = index * 7 + bit + (2 if index > 2 else 0)
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
    return state
