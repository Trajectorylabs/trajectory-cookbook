import numpy as np


HORIZON = 320
SCHOOL_INTERVAL = 8
MAX_FISH = (HORIZON // SCHOOL_INTERVAL) * 3
ANCHOR = (3, 0)


class CastNEnvironment:
  score_cap = MAX_FISH * (MAX_FISH + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.mode = 0
    self.power = 0
    self.water = 50
    self.steps = 0
    self.last_action = 0
    self.lure = ANCHOR
    self.fish: set[tuple[int, int, bool]] = set()
    self.caught: list[bool] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _line(self) -> set[tuple[int, int]]:
    row, column = self.lure
    if column == 0:
      return {ANCHOR}
    return {(round(ANCHOR[0] + (row - ANCHOR[0]) * x / column), x)
            for x in range(column + 1)}

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    if self.mode == 0 and pressed:
      self.mode, self.power = 1, 1
    elif self.mode == 1:
      if action == 5 and self.power < 3:
        self.power += 1
      else:
        self.lure = (2, self.power * 3)
        self.mode = 2
    elif self.mode == 2 and pressed:
      self.mode = 3
    if self.mode == 2:
      row, column = self.lure
      self.lure = (min(9, row + 1), column)

    if self.steps % 2 == 0:
      surface = min(8, self.water // 10 + 1)
      self.fish = {(max(surface, row), column - 1, red)
                   for row, column, red in self.fish if column > 0}
    if self.steps % SCHOOL_INTERVAL == 0:
      red = self.steps % (SCHOOL_INTERVAL * 4) == 0
      depth = min(7, self.water // 10 + 1 + int(red))
      row = int(self.random.integers(depth, 8))
      self.fish.update((row + offset, 9, red) for offset in range(3))

    reward = 0
    if self.mode == 3:
      line = self._line()
      hooked = {fish for fish in self.fish if fish[:2] in line}
      self.fish.difference_update(hooked)
      self.caught.extend(red for _, _, red in sorted(hooked))
      row, column = self.lure
      if column <= 1:
        for index, red in enumerate(self.caught, 1):
          reward += -1 if red else index
          self.water = max(0, min(50, self.water + (-5 if red else 1)))
        self.caught.clear()
        self.lure, self.mode = ANCHOR, 0
      else:
        self.lure = (row + int(np.sign(ANCHOR[0] - row)), column - 1)
    if self.steps % 2 == 0:
      self.water = max(0, self.water - 1)
    self.terminal = self.steps >= self.horizon or (self.mode != 3 and self.water < 26)
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.lure[0], self.lure[1], 0] = True
    for row, column, red in self.fish:
      state[row, column, 1 if red else 2] = True
    for row, column in self._line():
      state[row, column, 3] = True
    state[self.water // 10, :, 4] = True
    values = (self.mode, self.power, self.water, self.steps,
              len(self.caught), int(self.last_action == 5))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    for index, red in enumerate(self.caught):
      state[6 + index // 10, index % 10, 5] = red
    return state
