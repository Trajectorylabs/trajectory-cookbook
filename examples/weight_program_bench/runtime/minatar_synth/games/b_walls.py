from dataclasses import dataclass

import numpy as np


HORIZON = 400
COLUMN = 5
PLAYER_ROW = 9
PHASE_LENGTH = 6


@dataclass
class Wall:
  row: int
  phase: int
  width: int
  period: int
  direction: int

  def blocks(self, column: int) -> bool:
    return (column - self.phase) % self.period < self.width


class BWallsEnvironment:
  score_cap = HORIZON * 2 ** (PLAYER_ROW - 1)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.wall_clock = self.last_action = self.travel = 0
    self.multiplier = 1
    self.shot: int | None = None
    self.terminal = False
    self.walls: list[Wall] = []
    self._spawn(3)
    self._spawn(0)

  def _spawn(self, row: int) -> None:
    width = int(self.random.integers(1, 3))
    period = width + int(self.random.integers(2, 5))
    self.walls.append(Wall(row, int(self.random.integers(period)), width, period,
                           int(self.random.choice((-1, 1)))))

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  @property
  def stopped(self) -> bool:
    return self.wall_clock // PHASE_LENGTH % 2 == 1

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    if action == 5 and self.last_action != 5 and self.shot is None:
      self.shot = PLAYER_ROW
      self.multiplier = 1
    self.last_action = action
    self.steps += 1
    moving = not self.stopped
    if moving:
      for wall in self.walls:
        wall.phase = (wall.phase + wall.direction) % wall.period

    near_launch = self.shot is not None and self.shot >= 7
    if near_launch or self.steps % (5 if self.steps < 200 else 4) == 0:
      self.travel += 1
      for wall in self.walls:
        wall.row += 1
      if self.travel % 3 == 0:
        self._spawn(0)

    reward = 0
    if self.shot is not None:
      self.shot -= 1
      if self.shot < 0 or any(wall.row == self.shot and wall.blocks(COLUMN) for wall in self.walls):
        self.shot = None
      else:
        for wall in self.walls:
          if wall.row - 1 == self.shot:
            reward = self.multiplier
            self.multiplier *= 2
            self.walls.remove(wall)
            break
        if moving and self.shot < max((wall.row for wall in self.walls), default=-2) + 1:
          self.shot = None
    if self.shot is None:
      self.wall_clock += 1
    self.terminal = any(wall.row == PLAYER_ROW and wall.blocks(COLUMN) for wall in self.walls)
    self.walls = [wall for wall in self.walls if wall.row <= PLAYER_ROW]
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[PLAYER_ROW, COLUMN, 0] = True
    if self.shot is not None:
      state[self.shot, COLUMN, 3] = True
    for wall in self.walls:
      for column in range(10):
        state[wall.row, column, 1] = wall.blocks(column)
        state[wall.row, column, 4] = (column - wall.phase - wall.direction) % wall.period < wall.width
      if wall.row > 0:
        state[wall.row - 1, COLUMN, 2] = True
    for row, value in enumerate((self.steps, self.wall_clock % (2 * PHASE_LENGTH),
                                 self.travel % 3, self.multiplier, self.last_action)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
