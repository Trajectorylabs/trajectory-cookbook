import numpy as np


WAVE_QUOTAS = (10, 20, 30)
WAVE_VALUES = (100, 200, 300)
LEVEL_BONUSES = (400, 600, 1_000, 1_100, 1_400, 1_600, 2_000, 2_100)


class StarGunnerEnvironment:
  """Eight escalating three-wave invasions with bidirectional single-shot fire."""

  score_cap = 142_600.0

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
    self.row = 5
    self.column = 5
    self.previous_row = self.row
    self.previous_column = self.column
    self.facing = 1
    self.lives = 5
    self.level = 0
    self.wave = 0
    self.spawned = 0
    self.killed = 0
    self.next_enemy_id = 0
    self.enemies: list[dict[str, int]] = []
    self.lasers: list[list[int]] = []
    self.bombs: list[list[int]] = []
    self.previous_enemies: list[tuple[int, int]] = []
    self.fire_ready = True
    self.score = 0.0
    self.next_extra_life = 10_000
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._fill_formation()

  @property
  def active_limit(self) -> int:
    return self.wave + 1

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_row = self.row
    self.previous_column = self.column
    self.previous_enemies = [(enemy["row"], enemy["column"]) for enemy in self.enemies]
    if action != 5:
      self.fire_ready = True
    if action == 1:
      self.column = max(0, self.column - 1)
      self.facing = -1
    elif action == 2:
      self.row = max(0, self.row - 1)
    elif action == 3:
      self.column = min(9, self.column + 1)
      self.facing = 1
    elif action == 4:
      self.row = min(9, self.row + 1)
    elif action == 5 and self.fire_ready:
      self.lasers.append([self.row, self.column, self.facing])
      self.fire_ready = False
    self._move_lasers()
    self._move_enemies()
    self._move_bombs()
    reward = self._resolve_hits()
    self._resolve_player_collisions()
    if not self.terminal:
      self._fill_formation()
    if self.killed >= WAVE_QUOTAS[self.wave] and not self.enemies:
      reward += float(LEVEL_BONUSES[self.level])
      self._advance_wave()
    future_score = self.score + reward
    while future_score >= self.next_extra_life:
      self.lives = min(255, self.lives + 1)
      self.next_extra_life += 10_000
    self.score = future_score
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _fill_formation(self) -> None:
    quota = WAVE_QUOTAS[self.wave]
    while len(self.enemies) < self.active_limit and self.spawned < quota:
      enemy_id = self.next_enemy_id
      self.next_enemy_id += 1
      self.spawned += 1
      self.enemies.append(
        {
          "id": enemy_id,
          "row": int(self.random.randint(1, 9)),
          "column": 0 if self.random.rand() < 0.5 else 9,
          "forming": 2,
        }
      )

  def _move_lasers(self) -> None:
    for laser in self.lasers:
      laser[1] += laser[2]
    self.lasers = [laser for laser in self.lasers if 0 <= laser[1] < 10]

  def _move_enemies(self) -> None:
    interval = max(1, 4 - self.level // 2)
    for enemy in self.enemies:
      if enemy["forming"] > 0:
        enemy["forming"] -= 1
        continue
      if self.steps % interval == 0:
        enemy["column"] += int(np.sign(self.column - enemy["column"]))
        if self.steps % 2 == 0:
          enemy["row"] += int(np.sign(self.row - enemy["row"]))

  def _move_bombs(self) -> None:
    if self.steps % max(5, 11 - self.level) == 0:
      self.bombs.append([0, int(self.random.randint(10))])
    for bomb in self.bombs:
      bomb[0] += 1
    self.bombs = [bomb for bomb in self.bombs if bomb[0] < 10]

  def _resolve_hits(self) -> float:
    reward = 0.0
    hit_enemy_ids: set[int] = set()
    hit_lasers: set[int] = set()
    for laser_index, laser in enumerate(self.lasers):
      for enemy in self.enemies:
        if enemy["forming"] > 0 or laser[:2] != [enemy["row"], enemy["column"]]:
          continue
        hit_enemy_ids.add(enemy["id"])
        hit_lasers.add(laser_index)
        reward += float(WAVE_VALUES[self.wave])
        break
    self.enemies = [enemy for enemy in self.enemies if enemy["id"] not in hit_enemy_ids]
    self.lasers = [laser for index, laser in enumerate(self.lasers) if index not in hit_lasers]
    self.killed += len(hit_enemy_ids)
    return reward

  def _resolve_player_collisions(self) -> None:
    colliding_enemy_ids = {
      enemy["id"]
      for enemy in self.enemies
      if enemy["row"] == self.row and enemy["column"] == self.column
    }
    collision = bool(colliding_enemy_ids) or any(bomb == [self.row, self.column] for bomb in self.bombs)
    if not collision:
      return
    self.lives -= 1
    self.row = 5
    self.column = 5
    self.lasers.clear()
    self.bombs.clear()
    self.enemies = [enemy for enemy in self.enemies if enemy["id"] not in colliding_enemy_ids]
    self.spawned -= len(colliding_enemy_ids)
    if self.lives <= 0:
      self.terminal = True

  def _advance_wave(self) -> None:
    self.wave += 1
    if self.wave >= 3:
      self.wave = 0
      self.level += 1
    if self.level >= 8:
      self.terminal = True
      return
    self.spawned = 0
    self.killed = 0
    self._fill_formation()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.row, self.column, 0] = True
    state[self.previous_row, self.previous_column, 4] = True
    for enemy in self.enemies:
      channel = 2 if enemy["forming"] > 0 else 1
      state[enemy["row"], enemy["column"], channel] = True
    for row, column in self.previous_enemies:
      state[row, column, 4] = True
    for row, column, _ in self.lasers:
      state[row, column, 3] = True
    for row, column in self.bombs:
      state[row, column, 1] = True
    values = (
      (self.level, 4),
      (self.wave, 2),
      (self.spawned, 5),
      (self.killed, 5),
      (self.lives, 8),
      (self.row, 4),
      (self.column, 4),
      (self.facing + 1, 2),
      (self.steps, 13),
      (self.next_extra_life // 10_000, 5),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
      offset += width
    state[6, 0, 5] = self.fire_ready
    return state
