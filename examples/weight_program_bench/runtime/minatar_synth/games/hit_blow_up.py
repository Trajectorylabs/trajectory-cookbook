import numpy as np


class HitBlowUpEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.stage = self.action = 0
    self.terminal = False
    self._new_stage()

  def _new_stage(self) -> None:
    stage = self.stage % 6
    if stage == 0:
      self.palette = self.random.permutation(6).tolist()
    self.selector = self.palette[:3 + (stage + 1) // 2]
    self.target = self.random.permutation(self.selector)[:2 + stage // 2].tolist()
    self.cursor = self.cooldown = 0
    self.selector_y = 90.0
    self.guess: list[int] = []
    self.history: list[tuple[list[int], int, int]] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0.0
    if self.cooldown:
      self.cooldown -= 4
      if not self.cooldown:
        self.stage += 1
        self._new_stage()
    else:
      self.selector_y -= 0.2 * 2 ** (self.stage // 6) / (len(self.target) ** 2 + len(self.selector))
      if self.action == 1:
        self.cursor = (self.cursor - 1) % len(self.selector)
      elif self.action == 3:
        self.cursor = (self.cursor + 1) % len(self.selector)
      elif self.action == 5 and previous != 5:
        reward = self._select()
      self.terminal = not self.cooldown and self.selector_y - 3 - 6 * len(self.history) < 3
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _select(self) -> float:
    self.guess.append(self.selector[self.cursor])
    if len(self.guess) < len(self.target):
      return 0.0
    hit = sum(a == b for a, b in zip(self.target, self.guess, strict=True))
    blow = len(set(self.target).intersection(self.guess)) - hit
    self.history.append((self.guess, hit, blow))
    self.guess = []
    if hit == len(self.target):
      self.cooldown = 60
      return (self.selector_y - 6 * len(self.history)) * (self.stage // 6 + 1)
    return 0.0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for index, (guess, hit, blow) in enumerate(self.history):
      row, start = (index, 0) if index < 7 else (index - 7, 5)
      for column, color in enumerate(guess, start):
        state[row, column, color] = True
      for bit in range(3):
        state[row, start + 4, bit] = bool(hit & (1 << bit))
        state[row, start + 4, bit + 3] = bool(blow & (1 << bit))
    for column, color in enumerate(self.selector):
      state[7, column, color] = True
    if not self.cooldown:
      state[9, self.cursor, 4] = True
    for column in range(len(self.target)):
      if column < len(self.guess):
        state[8, column, self.guess[column]] = True
    for row, value in ((8, round(self.selector_y)), (9, self.stage)):
      for bit in range(6):
        state[row, 9, bit] = bool(value & (1 << bit))
      state[row, 8, 5] = bool(value & 64)
    state[9, 8, 0] = self.cooldown > 0
    state[9, 8, 1] = self.action == 5
    state[8, 5:5 + len(self.target), 0] = True
    if self.cooldown:
      for column, color in enumerate(self.target):
        state[9, column, color] = True
    return state
