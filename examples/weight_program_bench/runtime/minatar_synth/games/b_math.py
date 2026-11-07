import numpy as np


HORIZON = 400
FULL_TIME = 60
FEEDBACK_STEPS = 3
MAX_ANSWERS = (HORIZON + FEEDBACK_STEPS) // (FEEDBACK_STEPS + 1)


class BMathEnvironment:
  score_cap = 57 * MAX_ANSWERS * (MAX_ANSWERS + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.questions = self.feedback = self.last_action = 0
    self.time = FULL_TIME
    self.multiplier = 1
    self.correct = False
    self.selected = 0
    self.terminal = False
    self._question()

  def _question(self) -> None:
    self.operation = int(self.random.integers(min(4, 1 + self.questions // 5)))
    if self.operation < 2:
      total = int(self.random.integers(10, 100))
      part = int(self.random.integers(1, total))
      self.left, self.right, self.answer = (
        (part, total - part, total) if self.operation == 0 else (total, part, total - part)
      )
    else:
      factor = int(self.random.integers(2, 10))
      quotient = int(self.random.integers(2, 99 // factor + 1))
      product = factor * quotient
      self.left, self.right, self.answer = (
        (quotient, factor, product) if self.operation == 2 else (product, factor, quotient)
      )
    distractors = [value for value in range(max(1, self.answer - 7), min(99, self.answer + 7) + 1)
                   if value != self.answer]
    choices = self.random.choice(distractors, size=4, replace=False)
    self.options = self.random.permutation([self.answer, *choices]).tolist()

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
    reward = 0
    if self.feedback:
      self.feedback -= 1
      if not self.feedback:
        self._question()
    elif 1 <= action <= 5:
      reward = (57 * self.time * self.multiplier + FULL_TIME - 1) // FULL_TIME
      self.correct = self.options[action - 1] == self.answer
      self.selected = action
      if self.correct:
        self.time += 30
        if self.time >= FULL_TIME:
          self.multiplier += 1
      else:
        self.time -= 15
        self.multiplier = max(1, self.multiplier - 1)
      self.time = min(FULL_TIME, self.time)
      self.feedback = FEEDBACK_STEPS
      self.questions += 1
    self.time = max(0, self.time - 1)
    self.terminal = self.time == 0 or self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.operation, 0, 0] = True
    for row, value in enumerate((self.left, self.right, *self.options)):
      for bit in range(8):
        state[row, bit, 1] = bool(value & (1 << bit))
    if self.feedback:
      state[0, self.selected, 2] = True
      state[int(self.correct), 0, 3] = True
    for row, value in enumerate((self.time, self.multiplier, self.questions,
                                 self.feedback, self.steps, self.last_action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
