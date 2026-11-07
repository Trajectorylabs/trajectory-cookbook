import numpy as np


CUBES = tuple((row, column) for row in range(7) for column in range(row + 1))
DIRECTIONS = {1: (-1, -1), 2: (-1, 0), 3: (1, 1), 4: (1, 0)}


class QbertEnvironment:
  """Twenty-round color pyramid campaign with balls, Sam, Coily, and discs."""

  score_cap = 94_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 5_000):
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
    self.level = 0
    self.round = 0
    self.lives = 4
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.bonus_rounds: set[int] = set()
    self._start_round()

  def _start_round(self) -> None:
    self.player = [0, 0]
    self.previous_player = self.player.copy()
    self.colors = {cube: 0 for cube in CUBES}
    self.rewarded_cubes: set[tuple[int, int]] = set()
    self.discs = {"left", "right"}
    self.enemies = {
      "red": [0, 0],
      "purple": [0, 0],
      "sam": [0, 0],
      "green": [0, 0],
    }
    self.previous_enemies = {kind: position.copy() for kind, position in self.enemies.items()}
    self.caught_rewards: set[str] = set()
    self.freeze_timer = 0
    self.round_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_enemies = {kind: position.copy() for kind, position in self.enemies.items()}
    campaign_round_before = self.level * 4 + self.round
    reward = self._move_player(action)
    if self.level * 4 + self.round == campaign_round_before and not self.terminal:
      if self.freeze_timer == 0:
        self._move_enemies()
      reward += self._resolve_collisions()
    self.score += reward
    self.steps += 1
    self.round_steps += 1
    self.freeze_timer = max(0, self.freeze_timer - 1)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> float:
    if action not in DIRECTIONS:
      return 0.0
    row_delta, column_delta = DIRECTIONS[action]
    target = [self.player[0] + row_delta, self.player[1] + column_delta]
    if tuple(target) not in CUBES:
      disc = None
      if self.player == [3, 0] and action == 1:
        disc = "left"
      elif self.player == [3, 3] and action == 2:
        disc = "right"
      if disc in self.discs:
        self.discs.remove(disc)
        self.player = [0, 0]
        if "coily" in self.enemies:
          del self.enemies["coily"]
          if "coily" not in self.caught_rewards:
            self.caught_rewards.add("coily")
            return 500.0
        return 0.0
      self._lose_life(fell=True)
      return 0.0
    self.player = target
    reward = self._change_cube(tuple(target))
    if self._round_complete():
      reward += 3_100.0
      self._advance_round()
    return reward

  def _change_cube(self, cube: tuple[int, int]) -> float:
    color = self.colors[cube]
    if self.level == 0:
      color = 1
    elif self.level == 1:
      color = min(2, color + 1)
    elif self.level == 2:
      color = 0 if color == 1 else 1
    elif self.level == 3:
      color = 1 if color == 2 else color + 1
    else:
      color = (color + 1) % 3
    self.colors[cube] = color
    if color == self._destination_color() and cube not in self.rewarded_cubes:
      self.rewarded_cubes.add(cube)
      return 25.0
    return 0.0

  def _destination_color(self) -> int:
    return 1 if self.level in (0, 2) else 2

  def _round_complete(self) -> bool:
    destination = self._destination_color()
    return all(color == destination for color in self.colors.values())

  def _move_enemies(self) -> None:
    interval = max(1, 4 - self.level // 2)
    if self.round_steps % interval:
      return
    for kind in list(self.enemies):
      position = self.enemies[kind]
      if kind == "coily":
        self.enemies[kind] = self._chase_step(position)
        continue
      if position[0] >= 6:
        if kind == "purple":
          del self.enemies[kind]
          self.enemies["coily"] = [6, position[1]]
        else:
          del self.enemies[kind]
        continue
      choose_right = (self.round_steps + len(kind) + self.round) % 2
      position[0] += 1
      position[1] += choose_right
      if kind == "sam":
        self.colors[tuple(position)] = self._reverted_color(self.colors[tuple(position)])

  def _chase_step(self, position: list[int]) -> list[int]:
    candidates = []
    for row_delta, column_delta in DIRECTIONS.values():
      target = (position[0] + row_delta, position[1] + column_delta)
      if target in CUBES:
        distance = abs(target[0] - self.player[0]) + abs(target[1] - self.player[1])
        candidates.append((distance, target))
    return list(min(candidates)[1]) if candidates else position

  def _reverted_color(self, color: int) -> int:
    return max(0, color - 1)

  def _resolve_collisions(self) -> float:
    reward = 0.0
    for kind in list(self.enemies):
      if self.enemies[kind] != self.player:
        continue
      if kind == "sam":
        if kind not in self.caught_rewards:
          reward += 300.0
          self.caught_rewards.add(kind)
        del self.enemies[kind]
      elif kind == "green":
        if kind not in self.caught_rewards:
          reward += 100.0
          self.caught_rewards.add(kind)
        self.freeze_timer = 4
        del self.enemies[kind]
      else:
        self._lose_life(fell=False)
        break
    return reward

  def _lose_life(self, fell: bool) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    if fell:
      self.player = [0, 0]

  def _advance_round(self) -> None:
    completed = self.level * 4 + self.round + 1
    if completed == 5 or (completed > 5 and (completed - 5) % 4 == 0):
      self.lives += 1
      self.bonus_rounds.add(completed)
    self.round += 1
    if self.round >= 4:
      self.round = 0
      self.level += 1
    if self.level >= 5:
      self.terminal = True
    else:
      self._start_round()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.player[0] + 1, self.player[1] + 1, 0] = True
    state[self.previous_player[0] + 1, self.previous_player[1] + 1, 4] = True
    for kind, position in self.enemies.items():
      state[position[0] + 1, position[1] + 1, 1] = True
      if kind in self.previous_enemies:
        previous = self.previous_enemies[kind]
        state[previous[0] + 1, previous[1] + 1, 4] = True
    for cube, color in self.colors.items():
      row, column = cube
      state[row + 1, column + 1, 2] = color >= 1
      state[row + 1, column + 1, 3] = color >= 2
    for cube in self.rewarded_cubes:
      row, column = cube
      state[row + 1, column + 1, 5] = True
    state[4, 0, 3] = "left" in self.discs
    state[4, 5, 3] = "right" in self.discs
    cube_gauge_cells = {(row + 1, column + 1) for row, column in CUBES}
    free_cells = [(row, column) for row in range(10) for column in range(10) if (row, column) not in cube_gauge_cells]
    gauge_bits = []
    for bit in range(3):
      gauge_bits.append(bool(self.level & (1 << bit)))
    for bit in range(2):
      gauge_bits.append(bool(self.round & (1 << bit)))
    for bit in range(4):
      gauge_bits.append(bool(self.lives & (1 << bit)))
    for bit in range(13):
      gauge_bits.append(bool(self.steps & (1 << bit)))
    for bit in range(3):
      gauge_bits.append(bool(self.freeze_timer & (1 << bit)))
    gauge_bits.extend(kind in self.caught_rewards for kind in ("sam", "green", "coily"))
    gauge_bits.extend(("left" in self.discs, "right" in self.discs))
    for kind in ("red", "purple", "sam", "green", "coily"):
      position = self.enemies.get(kind)
      code = 31 if position is None else position[0] * 7 + position[1]
      for bit in range(5):
        gauge_bits.append(bool(code & (1 << bit)))
    for cell, value in zip(free_cells, gauge_bits):
      state[*cell, 5] = value
    return state
