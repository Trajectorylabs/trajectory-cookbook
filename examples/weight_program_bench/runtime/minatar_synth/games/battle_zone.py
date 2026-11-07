import numpy as np


DIRECTIONS = (
  (-1, 0),
  (-1, 1),
  (0, 1),
  (1, 1),
  (1, 0),
  (1, -1),
  (0, -1),
  (-1, -1),
)
POINTS = {"tank": 1_000.0, "fighter": 2_000.0, "supertank": 3_000.0, "saucer": 5_000.0}
WAVES = (
  ("tank", "tank", "tank", "fighter", "saucer"),
  ("tank", "tank", "fighter", "fighter", "saucer"),
  ("tank", "tank", "fighter", "supertank", "saucer"),
  ("tank", "fighter", "fighter", "supertank", "saucer"),
  ("tank", "fighter", "supertank", "supertank", "saucer"),
)
OBSTACLES = {(2, 2), (2, 7), (5, 1), (5, 8), (7, 3), (7, 6)}
SPAWNS = ((0, 1), (0, 5), (0, 9), (4, 0), (4, 9))


class BattleZoneEnvironment:
  """Radar combat with oriented movement, cover, shell dodging, and four target classes."""

  score_cap = 60_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_500):
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
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.direction = 0
    self.lives = 5
    self.wave = 0
    self.steps = 0
    self.total_score = 0.0
    self.bonus_awarded = False
    self.player_shell: list[int] | None = None
    self.enemy_shells: list[list[int]] = []
    self.last_action = 0
    self.terminal = False
    self._start_wave()

  def _start_wave(self) -> None:
    self.enemies = [
      {"kind": kind, "position": [*SPAWNS[index]], "previous": [*SPAWNS[index]]}
      for index, kind in enumerate(WAVES[self.wave])
    ]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    if action == 1:
      self.direction = (self.direction - 1) % 8
    elif action == 3:
      self.direction = (self.direction + 1) % 8
    elif action in (2, 4):
      multiplier = 1 if action == 2 else -1
      row_delta, column_delta = DIRECTIONS[self.direction]
      target = [self.player[0] + multiplier * row_delta, self.player[1] + multiplier * column_delta]
      if self._open(target):
        self.player = target
    elif action == 5 and self.player_shell is None:
      row_delta, column_delta = DIRECTIONS[self.direction]
      self.player_shell = [self.player[0], self.player[1], row_delta, column_delta, 9]

    self._move_enemies()
    self._move_shells()
    reward = self._resolve_player_shell()
    self._resolve_enemy_shells()
    self._enemy_fire()
    self.steps += 1
    if self.total_score >= 50_000 and not self.bonus_awarded:
      self.lives += 1
      self.bonus_awarded = True
    if not self.enemies:
      self.wave += 1
      if self.wave == len(WAVES):
        self.terminal = True
      else:
        self._start_wave()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  @staticmethod
  def _open(position: list[int]) -> bool:
    return 0 <= position[0] < 10 and 0 <= position[1] < 10 and tuple(position) not in OBSTACLES

  def _move_enemies(self) -> None:
    for enemy in self.enemies:
      enemy["previous"] = list(enemy["position"])
      kind = str(enemy["kind"])
      if kind == "saucer":
        enemy["position"][1] = (enemy["position"][1] + 1) % 10
        continue
      interval = {"tank": 3, "fighter": 2, "supertank": 1}[kind]
      if self.steps % interval != 0:
        continue
      position = list(enemy["position"])
      row_delta = int(np.sign(self.player[0] - position[0]))
      column_delta = int(np.sign(self.player[1] - position[1]))
      if kind == "fighter" and self.steps % 4 < 2:
        column_delta = -column_delta or 1
      candidates = ([position[0] + row_delta, position[1] + column_delta], [position[0] + row_delta, position[1]])
      for target in candidates:
        if self._open(target) and target != self.player:
          enemy["position"] = target
          break

  def _move_shells(self) -> None:
    if self.player_shell is not None:
      self.player_shell[0] += self.player_shell[2]
      self.player_shell[1] += self.player_shell[3]
      self.player_shell[4] -= 1
      if not self._open(self.player_shell[:2]) or self.player_shell[4] <= 0:
        self.player_shell = None
    remaining = []
    for shell in self.enemy_shells:
      shell[0] += shell[2]
      shell[1] += shell[3]
      shell[4] -= 1
      if self._open(shell[:2]) and shell[4] > 0:
        remaining.append(shell)
    self.enemy_shells = remaining

  def _resolve_player_shell(self) -> float:
    if self.player_shell is None:
      return 0.0
    hit = next((enemy for enemy in self.enemies if enemy["position"] == self.player_shell[:2]), None)
    if hit is None:
      return 0.0
    self.enemies.remove(hit)
    self.player_shell = None
    reward = POINTS[str(hit["kind"])]
    self.total_score += reward
    return reward

  def _resolve_enemy_shells(self) -> None:
    remaining = []
    for shell in self.enemy_shells:
      friendly_hit = next((enemy for enemy in self.enemies if enemy["position"] == shell[:2]), None)
      if friendly_hit is not None:
        self.enemies.remove(friendly_hit)
      elif shell[:2] == self.player:
        self._lose_tank()
      else:
        remaining.append(shell)
    self.enemy_shells = remaining

  def _enemy_fire(self) -> None:
    for enemy in self.enemies:
      kind = str(enemy["kind"])
      if kind == "saucer":
        continue
      interval = {"tank": 12, "fighter": 9, "supertank": 7}[kind]
      if self.steps % interval != 0:
        continue
      position = list(enemy["position"])
      row_delta = int(np.sign(self.player[0] - position[0]))
      column_delta = int(np.sign(self.player[1] - position[1]))
      self.enemy_shells.append([position[0], position[1], row_delta, column_delta, 10])

  def _lose_tank(self) -> None:
    self.lives -= 1
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.direction = 0
    self.player_shell = None
    self.enemy_shells.clear()
    if self.lives <= 0:
      self.terminal = True

  def _in_view(self, position: list[int]) -> bool:
    row_delta = int(np.sign(position[0] - self.player[0]))
    column_delta = int(np.sign(position[1] - self.player[1]))
    if (row_delta, column_delta) == (0, 0):
      return True
    target_direction = DIRECTIONS.index((row_delta, column_delta))
    return min((target_direction - self.direction) % 8, (self.direction - target_direction) % 8) <= 1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies:
      position = list(enemy["position"])
      state[*position, 1] = True
      if self._in_view(position):
        state[*position, 2] = True
      type_marker = {"tank": 0, "fighter": 1, "supertank": 2, "saucer": 3}[str(enemy["kind"])]
      state[position[0], type_marker, 5] = True
      state[*list(enemy["previous"]), 4] = True
    for shell in self.enemy_shells:
      state[shell[0], shell[1], 1] = True
    if self.player_shell is not None:
      state[self.player_shell[0], self.player_shell[1], 3] = True
    state[*self.previous_player, 4] = True
    for obstacle in OBSTACLES:
      state[*obstacle, 5] = True
    state[9, : min(10, self.lives), 5] = True
    state[8, : self.wave + 1, 5] = True
    state[7, self.direction, 5] = True
    return state
