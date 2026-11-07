import numpy as np


ORBIT = (5, 6, 7, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2, 3, 4)
HORIZON = 500


class BallTourEnvironment:
  # A ball is spawned no more often than every three columns of travel.
  score_cap = 999 * ((HORIZON + 2) // 3)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.distance = self.scroll_fraction = self.last_action = 0
    self.next_spike = 1
    self.next_ball = 2
    self.multiplier = 100
    self.spikes: list[tuple[int, int, int]] = []
    self.balls: list[tuple[int, int]] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  @property
  def row(self) -> int:
    return ORBIT[self.steps % len(ORBIT)]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    self.scroll_fraction += 10 if action == 5 else 1
    advance, self.scroll_fraction = divmod(self.scroll_fraction, 10)
    self.distance += advance
    moved_spikes = []
    for row, column, velocity in self.spikes:
      column += advance
      if column > 9:
        continue
      if velocity and self.steps % 4 == 0:
        if not 1 <= row + velocity <= 8:
          velocity = -velocity
        row += velocity
      moved_spikes.append((row, column, velocity))
    self.spikes = moved_spikes
    if self.distance == self.next_spike:
      row = int(self.random.integers(1, 9))
      velocity = int(self.random.choice((-1, 1))) if self.random.random() < 0.2 else 0
      self.spikes.append((row, 0, velocity))
      self.next_spike += int(self.random.integers(1, 6))
    self.terminal = any(column == 9 and row == self.row - 1 for row, column, _ in self.spikes)
    if self.terminal:
      return 0.0, True
    reward = 0
    remaining = []
    for row, column in self.balls:
      column += advance
      if column == 9 and row in (self.row - 1, self.row):
        reward += self.multiplier // 100
        self.multiplier = min(99_900, self.multiplier + 1_000)
      elif column <= 9:
        remaining.append((row, column))
    self.balls = remaining
    if self.distance == self.next_ball:
      row = int(self.random.integers(2, 8))
      if any(spike_row == row and column == 0 for spike_row, column, _ in self.spikes):
        self.next_ball += 1
      else:
        self.balls.append((row, 0))
        self.next_ball += int(self.random.integers(3, 8))
    self.multiplier = max(100, self.multiplier - 2)
    self.terminal = self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.row, 9, 0] = True
    state[self.row - 1, 9, 3] = True
    state[ORBIT[(self.steps + 1) % len(ORBIT)], 9, 4] = True
    for row, column in self.balls:
      state[row, column, 2] = True
    for row, column, velocity in self.spikes:
      state[row, column, 1] = True
      state[row, column, 4] = velocity == 1
      state[row, column, 5] = velocity == -1
    gauges = ((self.steps, 9), (self.scroll_fraction, 4), (self.multiplier, 17),
              (self.next_spike - self.distance, 3), (self.next_ball - self.distance, 3),
              (int(self.last_action == 5), 1))
    offset = 0
    for value, width in gauges:
      for bit in range(width):
        index = offset + bit
        state[index // 9, index % 9, 3] = bool(value & (1 << bit))
      offset += width
    return state
