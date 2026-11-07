from dataclasses import dataclass

import numpy as np


HORIZON = 400
WIDTH = 10
MAX_PANEL_WIDTH = 6
MAX_COINS = (HORIZON // 4) * MAX_PANEL_WIDTH


@dataclass
class Panel:
  row: int
  left: int
  width: int
  ladder: int
  depth: int
  has_coins: bool = False

  def floor(self) -> set[tuple[int, int]]:
    return {(self.row, column) for column in range(self.left, self.left + self.width)}

  def rungs(self) -> set[tuple[int, int]]:
    return {(row, self.left + self.ladder) for row in range(self.row, self.row + self.depth)}


class LadderDropEnvironment:
  score_cap = MAX_COINS * (MAX_COINS + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.panel_count = 0
    self.row, self.column = 8, 1
    self.direction = self.panel_direction = self.multiplier = 1
    self.climbing = self.falling = self.terminal = False
    self.panels = [Panel(9, 0, WIDTH, 0, 0)]
    self.coins: set[tuple[int, int]] = set()
    self._new_panel(5)

  def _new_panel(self, left: int) -> None:
    self.panel_count += 1
    width = int(self.random.integers(4, MAX_PANEL_WIDTH + 1))
    self.panel = Panel(0, min(left, WIDTH - width), width,
                       int(self.random.integers(width)), int(self.random.integers(3, 6)),
                       self.panel_count % 4 == 0)
    self.falling = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _settled_cells(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    floors: set[tuple[int, int]] = set()
    ladders: set[tuple[int, int]] = set()
    for panel in self.panels:
      floors.update(panel.floor())
      ladders.update(panel.rungs())
    return floors, ladders

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    if self.steps % 16 == 0 or (self.steps % 4 == 0 and min(p.row for p in self.panels) < 3):
      self.row += 1
      for panel in self.panels:
        panel.row += 1
      self.panels = [panel for panel in self.panels if panel.row < 10]
      self.multiplier = max(1, self.multiplier - sum(row == 9 for row, _ in self.coins))
      self.coins = {(row + 1, column) for row, column in self.coins if row < 9}

    floors, ladders = self._settled_cells()
    if not self.falling:
      left = self.panel.left + self.panel_direction
      if not 0 <= left <= WIDTH - self.panel.width:
        self.panel_direction *= -1
      else:
        self.panel.left = left
      self.falling = pressed
    if self.falling:
      self.panel.row += 1
      if (self.row, self.column) in self.panel.floor() - self.panel.rungs():
        self.terminal = True
        return 0.0, True
      if (self.panel.floor() | self.panel.rungs()) & (floors | ladders):
        self.panel.row -= 1
        self.panels.append(self.panel)
        if self.panel.has_coins:
          self.coins.update((self.panel.row - 1, column)
                            for column in range(self.panel.left, self.panel.left + self.panel.width)
                            if self.panel.row > 0)
        self._new_panel(self.panel.left)
        floors, ladders = self._settled_cells()
      elif self.panel.row >= 10:
        self._new_panel(self.panel.left)

    self._move_player(floors, ladders)
    reward = 0
    cell = self.row, self.column
    if cell in self.coins:
      self.coins.remove(cell)
      reward = self.multiplier
      self.multiplier += 1
    self.terminal = self.row >= 10 or self.steps >= self.horizon or not self.panels
    return float(reward), self.terminal

  def _move_player(self, floors: set, ladders: set) -> None:
    if (self.row, self.column) in ladders:
      self.row -= 1
      self.climbing = True
      return
    if (self.row + 1, self.column) not in floors:
      self.row += 1
      self.climbing = False
      return
    destination = self.column + self.direction
    if not 0 <= destination < WIDTH or (self.row, destination) in floors - ladders:
      self.direction *= -1
    else:
      self.column = destination
    self.climbing = False

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, self.row)), self.column, 0] = True
    floors, ladders = self._settled_cells()
    for channel, cells in enumerate((floors, ladders, self.coins,
                                     self.panel.floor() | self.panel.rungs()), start=1):
      for row, column in cells:
        if 0 <= row < 10:
          state[row, column, channel] = True
    gauges = (self.steps, self.multiplier, self.panel_count % 4, self.panel.ladder,
              int(self.falling), int(self.direction > 0), int(self.panel_direction > 0),
              int(self.last_action == 5), self.row + 1)
    for row, value in enumerate(gauges):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
