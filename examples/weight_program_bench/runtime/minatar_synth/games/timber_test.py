import numpy as np


HORIZON = 400
UNIT = 8
MIN_WIDTH = 32
MAX_PIECES = 4
MAX_SPEED = 4
SAW_LEAD = 6


class TimberTestEnvironment:
  score_cap = HORIZON // ((MIN_WIDTH + SAW_LEAD + MAX_SPEED - 1) // MAX_SPEED) * (MAX_PIECES - 1) * 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.round = self.last_action = 0
    self.terminal = False
    self._new_log()

  def _new_log(self) -> None:
    self.round += 1
    width = int(self.random.integers(MIN_WIDTH, 65))
    slack = 80 - width
    self.start = slack // 2 + int(self.random.integers(slack // 3 + 1))
    self.end = self.start + width
    self.remaining = self.start
    self.target = int(self.random.integers(2, MAX_PIECES + 1))
    self.saw = self.start - SAW_LEAD
    self.pieces: list[int] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _grade(self) -> int:
    score = (self.target - 1) * 100
    widths = self.pieces + [0] * max(0, self.target - len(self.pieces))
    for index, (left, right) in enumerate(zip(widths, widths[1:]), 1):
      if index > self.target or left + right == 0:
        score -= 100
      else:
        score -= abs(left - right) * 300 // (left + right)
    return score

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    self.saw = min(self.end, self.saw + min(MAX_SPEED, 1 + (self.round - 1) // 3))
    reward = 0
    if self.saw == self.end:
      self.pieces.append(self.end - self.remaining)
      score = self._grade()
      if score < 0:
        self.terminal = True
      else:
        reward = score
        if self.steps < self.horizon:
          self._new_log()
    elif pressed and self.saw > self.remaining:
      self.pieces.append(self.saw - self.remaining)
      self.remaining = self.saw
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[1, max(0, min(9, self.saw // UNIT)), 0] = True
    state[2, self.remaining // UNIT:(self.end - 1) // UNIT + 1, 1] = True
    if self.round <= 3 and not self.pieces:
      for index in range(1, self.target):
        cut = self.start + round((self.end - self.start) * index / self.target)
        state[3, cut // UNIT, 2] = True
    cut = self.start
    for width in self.pieces:
      cut += width
      index = cut - 1
      state[index // 10, index % 10, 3] = True
    values = (self.steps, self.round, self.target, self.start, self.end,
              self.remaining, self.saw + SAW_LEAD, int(self.last_action == 5))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
