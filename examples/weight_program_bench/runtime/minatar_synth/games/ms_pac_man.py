import numpy as np


FRUIT_VALUES = (100, 200, 500, 700)
GHOST_VALUES = (200, 400, 800, 1_600)
ENERGIZERS = ((1, 1), (1, 8), (8, 1), (8, 8))
PLAYER_START = (7, 5)
GHOST_STARTS = ((4, 4), (4, 5), (5, 4), (5, 5))
INTERNAL_WALLS = (
  frozenset({(2, 2), (2, 3), (2, 6), (2, 7), (7, 2), (7, 3), (7, 6), (7, 7)}),
  frozenset({(2, 4), (2, 5), (4, 2), (5, 2), (4, 7), (5, 7), (7, 4), (7, 5)}),
  frozenset({(2, 2), (2, 7), (3, 4), (3, 5), (6, 4), (6, 5), (7, 2), (7, 7)}),
  frozenset({(2, 3), (2, 6), (4, 2), (4, 7), (5, 2), (5, 7), (7, 3), (7, 6)}),
)
DIRECTIONS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class MsPacManEnvironment:
  """Four-maze dot clearing with fruit, energizers, and four ghosts."""

  score_cap = 52_180.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 4_000):
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
    self.maze = 0
    self.lives = 3
    self.score = 0.0
    self.bonus_life_awarded = False
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_maze()

  def _walls(self) -> set[tuple[int, int]]:
    walls = {(0, column) for column in range(10)} | {(9, column) for column in range(10)}
    walls |= {(row, 0) for row in range(10) if row != 5}
    walls |= {(row, 9) for row in range(10) if row != 5}
    walls |= set(INTERNAL_WALLS[self.maze])
    return walls

  def _start_maze(self) -> None:
    walls = self._walls()
    excluded = {PLAYER_START, *GHOST_STARTS, *ENERGIZERS, (5, 5)}
    self.dots = {
      (row, column)
      for row in range(1, 9)
      for column in range(1, 9)
      if (row, column) not in walls and (row, column) not in excluded
    }
    self.initial_dot_count = len(self.dots)
    self.energizers = set(ENERGIZERS)
    self.fruit_collected = False
    self.fruit_active = False
    self.fruit_timer = 0
    self.frightened_timer = 0
    self.ghost_chain = 0
    self.frightened_eaten: set[int] = set()
    self._reset_actors()

  def _reset_actors(self) -> None:
    self.player = list(PLAYER_START)
    self.previous_player = self.player.copy()
    self.ghosts = [list(position) for position in GHOST_STARTS]
    self.previous_ghosts = [ghost.copy() for ghost in self.ghosts]
    self.ghost_directions = [1, 3, 1, 3]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_ghosts = [ghost.copy() for ghost in self.ghosts]
    maze_before = self.maze
    self._move_player(action)
    reward = self._eat_at_player()
    if self.maze == maze_before and not self.terminal:
      self._move_ghosts()
      reward += self._resolve_collisions()
    self.score += reward
    if self.score >= 10_000 and not self.bonus_life_awarded:
      self.lives += 1
      self.bonus_life_awarded = True
    self.steps += 1
    self.frightened_timer = max(0, self.frightened_timer - 1)
    self.fruit_timer = max(0, self.fruit_timer - 1)
    if self.fruit_timer == 0:
      self.fruit_active = False
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if action not in DIRECTIONS:
      return
    row_delta, column_delta = DIRECTIONS[action]
    target = [self.player[0] + row_delta, self.player[1] + column_delta]
    if self.player[0] == 5 and target[1] < 0:
      target[1] = 9
    elif self.player[0] == 5 and target[1] > 9:
      target[1] = 0
    if tuple(target) not in self._walls():
      self.player = target

  def _eat_at_player(self) -> float:
    position = tuple(self.player)
    reward = 0.0
    if position in self.dots:
      self.dots.remove(position)
      reward += 10.0
    if position in self.energizers:
      self.energizers.remove(position)
      self.frightened_timer = max(8, 20 - self.maze * 4)
      self.ghost_chain = 0
      self.frightened_eaten.clear()
      reward += 50.0
    if not self.fruit_collected and not self.fruit_active and len(self.dots) <= self.initial_dot_count // 2:
      self.fruit_active = True
      self.fruit_timer = 80
    if self.fruit_active and position == (5, 5):
      self.fruit_active = False
      self.fruit_collected = True
      reward += float(FRUIT_VALUES[self.maze])
    if not self.dots and not self.energizers:
      self.maze += 1
      if self.maze >= 4:
        self.terminal = True
      else:
        self._start_maze()
    return reward

  def _move_ghosts(self) -> None:
    if self.maze == 0 and self.steps % 2:
      return
    walls = self._walls()
    for index, ghost in enumerate(self.ghosts):
      options = []
      reverse = {1: 3, 2: 4, 3: 1, 4: 2}[self.ghost_directions[index]]
      for action, (row_delta, column_delta) in DIRECTIONS.items():
        target = [ghost[0] + row_delta, ghost[1] + column_delta]
        if ghost[0] == 5 and target[1] < 0:
          target[1] = 9
        elif ghost[0] == 5 and target[1] > 9:
          target[1] = 0
        if tuple(target) not in walls and action != reverse:
          options.append((action, target))
      if not options:
        row_delta, column_delta = DIRECTIONS[reverse]
        options = [(reverse, [ghost[0] + row_delta, ghost[1] + column_delta])]
      target_player = self._ghost_target(index)
      scored = []
      for option in options:
        distance = abs(option[1][0] - target_player[0]) + abs(option[1][1] - target_player[1])
        scored.append((distance, option))
      best_value = max(value for value, _ in scored) if self.frightened_timer else min(value for value, _ in scored)
      best = [option for value, option in scored if value == best_value]
      action, target = best[int(self.random.randint(len(best)))]
      self.ghost_directions[index] = action
      self.ghosts[index] = target

  def _ghost_target(self, index: int) -> tuple[int, int]:
    if index == 0:
      return tuple(self.player)
    if index == 1:
      delta = DIRECTIONS.get(self.last_action, (0, 0))
      return self.player[0] + delta[0] * 2, self.player[1] + delta[1] * 2
    if index == 2:
      return (1, 1) if (self.steps // 8) % 2 else tuple(self.player)
    return (8, 8) if self.player[0] < 5 else (1, 8)

  def _resolve_collisions(self) -> float:
    reward = 0.0
    for index, ghost in enumerate(self.ghosts):
      crossed = ghost == self.previous_player and self.player == self.previous_ghosts[index]
      if ghost != self.player and not crossed:
        continue
      if self.frightened_timer and index not in self.frightened_eaten:
        reward += float(GHOST_VALUES[min(3, self.ghost_chain)])
        self.ghost_chain += 1
        self.frightened_eaten.add(index)
        self.ghosts[index] = list(GHOST_STARTS[index])
        self.previous_ghosts[index] = self.ghosts[index].copy()
      else:
        self.lives -= 1
        if self.lives <= 0:
          self.terminal = True
        else:
          self.frightened_timer = 0
          self.ghost_chain = 0
          self.frightened_eaten.clear()
          self._reset_actors()
        break
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    state[*self.previous_player, 4] = True
    for index, ghost in enumerate(self.ghosts):
      state[*ghost, 1] = True
      state[*self.previous_ghosts[index], 4] = True
    for dot in self.dots:
      state[*dot, 2] = True
    for energizer in self.energizers:
      state[*energizer, 2] = True
      state[*energizer, 3] = True
    if self.fruit_active:
      state[5, 5, 2] = True
      state[5, 5, 3] = True
    for wall in self._walls():
      state[*wall, 3] = True
    for bit in range(2):
      state[0, bit, 5] = bool(self.maze & (1 << bit))
    for bit in range(3):
      state[0, 2 + bit, 5] = bool(self.lives & (1 << bit))
      state[0, 5 + bit, 5] = bool(self.ghost_chain & (1 << bit))
    for bit in range(5):
      index = 8 + bit
      state[index // 10, index % 10, 5] = bool(self.frightened_timer & (1 << bit))
    for bit in range(7):
      index = 13 + bit
      state[index // 10, index % 10, 5] = bool(self.fruit_timer & (1 << bit))
    for index, direction in enumerate(self.ghost_directions):
      for bit in range(3):
        gauge_index = 20 + index * 3 + bit
        state[gauge_index // 10, gauge_index % 10, 5] = bool(direction & (1 << bit))
    for bit in range(12):
      gauge_index = 32 + bit
      state[gauge_index // 10, gauge_index % 10, 5] = bool(self.steps & (1 << bit))
    state[4, 4, 5] = self.fruit_collected
    state[4, 5, 5] = self.bonus_life_awarded
    for index in range(4):
      state[4, 6 + index, 5] = index in self.frightened_eaten
    return state
