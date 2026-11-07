import numpy as np


GATE_OFFSETS = tuple(2 + index * 2 for index in range(50))
GATE_COLUMNS = tuple(1 + (index * 3 + 2) % 8 for index in range(50))
OBSTACLES = tuple((3 + index * 4, (index * 5 + 1) % 10) for index in range(24))


class SkiingEnvironment:
  """Olympic slalom with ski-angle speed, gates, trees, and timed recovery."""

  score_cap = 6_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 300):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = horizon
    self.random = np.random.RandomState()
    self.seed(0)
    self.reset()

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def seed(self, seed: int) -> None:
    self.random.seed(seed)

  def reset(self) -> None:
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._restart_run()

  def _restart_run(self) -> None:
    self.progress = 0
    self.column = 5
    self.previous_column = self.column
    self.angle = 0
    self.previous_angle = self.angle
    self.recovery = 0
    self.score = 0.0
    self.passed: set[int] = set()
    self.missed: set[int] = set()
    self.crashed: set[int] = set()

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_column = self.column
    self.previous_angle = self.angle
    reward = 0.0
    if action == 5:
      self._restart_run()
    elif self.recovery > 0:
      self.recovery -= 1
    else:
      if action == 1:
        self.angle = max(-3, self.angle - 1)
      elif action == 3:
        self.angle = min(3, self.angle + 1)
      elif action == 2:
        self.angle -= int(np.sign(self.angle))
      elif action == 4:
        self.angle = -3 if self.angle <= 0 else 3
      progress_before = self.progress
      speed = 2 if self.angle == 0 else 1 if abs(self.angle) <= 2 else 0
      self.progress = min(100, self.progress + speed)
      self.column = min(9, max(0, self.column + int(np.sign(self.angle))))
      reward += self._resolve_gates(progress_before)
      self._resolve_obstacles(progress_before)
      if self.progress >= 100:
        reward += 1_000.0
        self.terminal = True
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _resolve_gates(self, previous_progress: int) -> float:
    reward = 0.0
    for index, (offset, gate_column) in enumerate(zip(GATE_OFFSETS, GATE_COLUMNS, strict=True)):
      if index in self.passed or index in self.missed or not previous_progress < offset <= self.progress:
        continue
      if abs(self.column - gate_column) <= 1:
        self.passed.add(index)
        reward += 100.0
      else:
        self.missed.add(index)
    return reward

  def _resolve_obstacles(self, previous_progress: int) -> None:
    for index, (offset, column) in enumerate(OBSTACLES):
      if index in self.crashed or not previous_progress < offset <= self.progress:
        continue
      if self.column == column:
        self.crashed.add(index)
        self.recovery = 3

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[2, self.column, 0] = True
    state[2, self.previous_column, 4] = True
    for index, (offset, gate_column) in enumerate(zip(GATE_OFFSETS, GATE_COLUMNS, strict=True)):
      row = 2 + offset - self.progress
      if 0 <= row < 10 and index not in self.passed and index not in self.missed:
        state[row, max(0, gate_column - 1), 2] = True
        state[row, min(9, gate_column + 1), 2] = True
    for index, (offset, column) in enumerate(OBSTACLES):
      row = 2 + offset - self.progress
      if 0 <= row < 10 and index not in self.crashed:
        state[row, column, 1] = True
    for index in range(50):
      state[index // 10, index % 10, 5] = index in self.passed
      state[5 + index // 10, index % 10, 5] = index in self.missed
    values = (
      (self.progress, 7),
      (self.column, 4),
      (self.angle + 3, 3),
      (self.previous_angle + 3, 3),
      (self.recovery, 2),
      (self.steps, 9),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 4] = bool(value & (1 << bit))
      offset += width
    return state
