import numpy as np


HORIZON = 300
MIN_PULSE_INTERVAL = 2
MAX_SPAWNS = (HORIZON + MIN_PULSE_INTERVAL - 1) // MIN_PULSE_INTERVAL
RAYS = (
  ((4, 6), (4, 7), (4, 8), (4, 9)),
  ((6, 5), (7, 6), (8, 7), (9, 8)),
  ((6, 3), (7, 2), (8, 1), (9, 0)),
  ((4, 3), (4, 2), (4, 1), (4, 0)),
  ((3, 3), (2, 2), (1, 1), (0, 0)),
  ((3, 5), (2, 6), (1, 7), (0, 8)),
)


class HexminEnvironment:
  score_cap = MAX_SPAWNS * (1 + 3 * (MAX_SPAWNS - 1))

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.lanes = [0] * len(RAYS)
    self.orientation = self.steps = self.last_action = self.repeat = self.spawn_wait = 0
    self.pulse = 1
    self.multiplier = 1
    self.terminal = False

  @property
  def gates(self) -> tuple[int, int]:
    return self.orientation, (self.orientation + 2) % len(RAYS)

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.steps += 1
    self.repeat = max(0, self.repeat - 1)
    if action in (1, 3, 5) and (action != self.last_action or self.repeat == 0):
      self.orientation = (self.orientation + (-1 if action == 1 else 1)) % len(RAYS)
      self.multiplier = max(1, (9 * self.multiplier + 9) // 10 - 1)
      self.repeat = 2
    self.last_action = action

    self.pulse -= 1
    if self.pulse == 0:
      self.lanes = [age + 1 if age else 0 for age in self.lanes]
      if max(self.lanes) > 4:
        self.terminal = True
        return 0.0, True
      if self.spawn_wait == 0:
        start = int(self.random.integers(len(RAYS)))
        for offset in range(len(RAYS)):
          lane = (start + offset) % len(RAYS)
          if self.lanes[lane] == 0:
            self.lanes[lane] = 1
            break
        self.spawn_wait = int(self.random.integers(max(1, 4 - self.steps // 100)))
      else:
        self.spawn_wait -= 1
      self.pulse = max(MIN_PULSE_INTERVAL, 4 - self.steps // 100)

    reward = 0
    for lane in range(len(RAYS)):
      if self.lanes[lane] == 4 and lane in self.gates:
        reward += self.multiplier
        self.multiplier += 6
        self.lanes[lane] = 0
    self.terminal = self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[4, 4, 0] = True
    for lane, cells in enumerate(RAYS):
      row, column = cells[0]
      state[row, column, 3] = lane in self.gates
      for row, column in cells[4 - min(self.lanes[lane], 4):]:
        state[row, column, 1] = True
      if self.lanes[lane] >= 4:
        row, column = cells[0]
        state[row, column, 2] = True
    gauges = (self.orientation, self.steps, self.pulse, self.spawn_wait,
              self.repeat, self.multiplier, self.last_action)
    for row, value in enumerate(gauges):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
