from dataclasses import dataclass

import numpy as np


HORIZON = 400
MAX_JUMP = 8
CENTER = 5


@dataclass(frozen=True)
class Lane:
  direction: int
  period: int
  phase: int
  width: int

  def traffic(self, tick: int) -> np.ndarray:
    if self.direction == 0:
      return np.zeros(10, dtype=bool)
    return (np.arange(10) - self.phase - self.direction * tick) % self.period < self.width


class FroooogEnvironment:
  score_cap = HORIZON * MAX_JUMP

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.position = self.bottom = self.charge = self.last_action = 0
    self.destination: int | None = None
    self.launch_position = 0
    self.terminal = False
    self.lanes = [Lane(0, 1, 0, 0) for _ in range(3)]
    self.next_safe = int(self.random.integers(9, 16))
    self._extend()

  def _extend(self) -> None:
    while len(self.lanes) < self.bottom + 10:
      if len(self.lanes) == self.next_safe:
        self.lanes.append(Lane(0, 1, 0, 0))
        self.next_safe += int(self.random.integers(9, 16))
      else:
        period = int(self.random.integers(5, 11))
        self.lanes.append(Lane(int(self.random.choice((-1, 1))), period,
                               int(self.random.integers(period)), int(self.random.integers(1, 3))))

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
    if self.destination is None:
      if self.lanes[self.position].traffic(self.steps)[CENTER]:
        self.terminal = True
        return 0.0, True
      if action == 5:
        self.charge += 1
      if self.charge and (action != 5 or self.charge == MAX_JUMP):
        self.launch_position = self.position
        self.destination = self.position + min(self.charge, self.bottom + 9 - self.position)
        self.charge = 0
    if self.destination is not None:
      self.position = min(self.position + 2, self.destination)
      if self.position == self.destination:
        landing = self.lanes[self.position]
        self.terminal = bool(landing.traffic(self.steps)[CENTER])
        if not self.terminal and landing.direction and self.lanes[self.launch_position].direction:
          reward = (self.position - self.launch_position) ** 2
        self.destination = None
    self.bottom = max(self.bottom, self.position - 1)
    if self.steps % 8 == 0:
      self.bottom += 1
    self._extend()
    self.terminal |= self.position < self.bottom or self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    if self.position >= self.bottom:
      state[9 - (self.position - self.bottom), CENTER, 0] = True
    for row, lane in enumerate(reversed(self.lanes[self.bottom:self.bottom + 10])):
      state[row, :, 1] = lane.traffic(self.steps)
      state[row, :, 2] = lane.direction == 0
      state[row, :, 4] = lane.traffic(self.steps + 1)
    if self.destination is not None and self.destination >= self.bottom:
      state[9 - (self.destination - self.bottom), CENTER, 3] = True
    elif self.charge:
      preview = min(9, self.position - self.bottom + self.charge)
      state[9 - preview, CENTER, 3] = True
    values = (self.steps, self.charge, int(self.destination is not None),
              self.position - self.launch_position if self.destination is not None else 0)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
