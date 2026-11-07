import numpy as np


ACTIONS = {
  1: (0, -1),
  2: (-1, 0),
  3: (0, 1),
  4: (1, 0),
}


class AmidarEnvironment:
  """Trace a rectilinear maze while avoiding deterministic patrol enemies."""

  score_cap = 9.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 600):
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
    self.player = [9, 0]
    self.previous_player = self.player.copy()
    self.enemies = [[0, 0], [0, 9], [9, 9]]
    self.previous_enemies = [position.copy() for position in self.enemies]
    self.enemy_directions = [4, 1, 2]
    self.painted_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    self.completed_boxes: set[tuple[int, int]] = set()
    self.shadow_timer = 0
    self.shadow_uses = 3
    self.last_action = 0
    self.steps = 0
    self.terminal = False

  @staticmethod
  def _on_path(position: list[int] | tuple[int, int]) -> bool:
    row, column = position
    return 0 <= row < 10 and 0 <= column < 10 and (row % 3 == 0 or column % 3 == 0)

  @staticmethod
  def _edge(first: list[int], second: list[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((tuple(first), tuple(second))))

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    if action == 5 and self.shadow_uses > 0 and self.shadow_timer == 0:
      self.shadow_uses -= 1
      self.shadow_timer = 12
    elif action in ACTIONS:
      row_delta, column_delta = ACTIONS[action]
      target = [self.player[0] + row_delta, self.player[1] + column_delta]
      if self._on_path(target):
        self.player = target
        if self.player != self.previous_player:
          self.painted_edges.add(self._edge(self.previous_player, self.player))

    reward = self._complete_boxes()
    self.previous_enemies = [position.copy() for position in self.enemies]
    if self.shadow_timer == 0:
      for index in range(len(self.enemies)):
        self._move_enemy(index)
    if self.shadow_timer == 0 and any(enemy == self.player for enemy in self.enemies):
      self.terminal = True
    self.shadow_timer = max(0, self.shadow_timer - 1)
    self.steps += 1
    if len(self.completed_boxes) == 9 or self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_enemy(self, index: int) -> None:
    direction = self.enemy_directions[index]
    preferences = (direction, {1: 2, 2: 3, 3: 4, 4: 1}[direction], {1: 4, 2: 1, 3: 2, 4: 3}[direction])
    for candidate_direction in preferences:
      row_delta, column_delta = ACTIONS[candidate_direction]
      target = [self.enemies[index][0] + row_delta, self.enemies[index][1] + column_delta]
      if self._on_path(target):
        self.enemies[index] = target
        self.enemy_directions[index] = candidate_direction
        return

  def _complete_boxes(self) -> float:
    reward = 0.0
    for box_row in range(3):
      for box_column in range(3):
        box = (box_row, box_column)
        if box in self.completed_boxes:
          continue
        top = box_row * 3
        left = box_column * 3
        required = set()
        for offset in range(3):
          required.add(self._edge([top, left + offset], [top, left + offset + 1]))
          required.add(self._edge([top + 3, left + offset], [top + 3, left + offset + 1]))
          required.add(self._edge([top + offset, left], [top + offset + 1, left]))
          required.add(self._edge([top + offset, left + 3], [top + offset + 1, left + 3]))
        if required <= self.painted_edges:
          self.completed_boxes.add(box)
          reward += 1.0
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies:
      channel = 3 if self.shadow_timer > 0 else 1
      state[*enemy, channel] = True
    for box_row in range(3):
      for box_column in range(3):
        center = (min(9, box_row * 3 + 1), min(9, box_column * 3 + 1))
        if (box_row, box_column) not in self.completed_boxes:
          state[*center, 2] = True
    for first, second in self.painted_edges:
      state[*first, 4] = True
      state[*second, 4] = True
    state[9, : self.shadow_uses, 5] = True
    if self.shadow_timer > 0:
      state[8, : min(10, self.shadow_timer), 5] = True
    return state
