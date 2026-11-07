import numpy as np


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
ERAS = (1910, 1940, 1970, 1983, 2001)
FORMATION_IDS = frozenset(range(20, 25))


class TimePilotEnvironment:
  """One five-era loop with scrolling flight, formations, and bosses."""

  score_cap = 59_000.0

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
    self.era = 0
    self.heading = 0
    self.previous_heading = self.heading
    self.lives = 3
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.next_extra_life = 10_000
    self.terminal = False
    self._start_era()

  def _start_era(self) -> None:
    self.spawned = 0
    self.killed = 0
    self.special_spawned = self.era == 0
    self.special_destroyed = self.era == 0
    self.formation_destroyed: set[int] = set()
    self.formation_failed = False
    self.enemies: list[dict[str, int | str]] = []
    self.bullets: list[list[int]] = []
    self.enemy_shots: list[list[int]] = []
    self.boss: dict[str, int] | None = None
    self.previous_objects: list[tuple[int, int]] = []
    self._spawn_enemies()

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_heading = self.heading
    self.previous_objects = [(int(enemy["row"]), int(enemy["column"])) for enemy in self.enemies]
    if self.boss is not None:
      self.previous_objects.append((self.boss["row"], self.boss["column"]))
    if action == 1:
      self.heading = (self.heading - 1) % 8
    elif action == 2:
      self._scroll_world(1)
    elif action == 3:
      self.heading = (self.heading + 1) % 8
    elif action == 4:
      self.heading = (self.heading + 4) % 8
    elif action == 5:
      row_delta, column_delta = DIRECTIONS[self.heading]
      self.bullets.append([5 + row_delta, 5 + column_delta, row_delta, column_delta])
    self._move_bullets()
    self._move_enemies()
    self._move_enemy_shots()
    reward = self._resolve_hits()
    self._resolve_collisions()
    if not self.terminal:
      self._spawn_enemies()
      self._spawn_boss_if_ready()
    future_score = self.score + reward
    while future_score >= self.next_extra_life:
      self.lives += 1
      self.next_extra_life += 50_000 if self.next_extra_life >= 10_000 else 10_000
    self.score = future_score
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _scroll_world(self, distance: int) -> None:
    row_delta, column_delta = DIRECTIONS[self.heading]
    for enemy in self.enemies:
      enemy["row"] -= row_delta * distance
      enemy["column"] -= column_delta * distance
    if self.boss is not None:
      self.boss["row"] -= row_delta * distance
      self.boss["column"] -= column_delta * distance

  def _spawn_enemies(self) -> None:
    if self.killed >= 56:
      return
    active_normal = sum(enemy["kind"] == "normal" for enemy in self.enemies)
    target_active = 3 + self.era // 2
    if self.spawned == 20 and not FORMATION_IDS.intersection(
      int(enemy["id"]) for enemy in self.enemies
    ):
      for enemy_id in FORMATION_IDS:
        self.enemies.append(self._new_enemy(enemy_id, "normal"))
      self.spawned = 25
      return
    while active_normal < target_active and self.spawned < 56 and self.spawned not in FORMATION_IDS:
      self.enemies.append(self._new_enemy(self.spawned, "normal"))
      self.spawned += 1
      active_normal += 1
    if self.era > 0 and not self.special_spawned and self.killed >= 10:
      self.enemies.append(self._new_enemy(100 + self.era, "special"))
      self.special_spawned = True

  def _new_enemy(self, enemy_id: int, kind: str) -> dict[str, int | str]:
    edge = int(self.random.randint(4))
    coordinate = int(self.random.randint(1, 9))
    row, column = ((0, coordinate), (9, coordinate), (coordinate, 0), (coordinate, 9))[edge]
    return {"id": enemy_id, "kind": kind, "row": row, "column": column}

  def _move_bullets(self) -> None:
    for bullet in self.bullets:
      bullet[0] += bullet[2]
      bullet[1] += bullet[3]
    self.bullets = [bullet for bullet in self.bullets if 0 <= bullet[0] < 10 and 0 <= bullet[1] < 10]

  def _move_enemies(self) -> None:
    interval = max(1, 4 - self.era // 2)
    if self.steps % interval == 0:
      for enemy in self.enemies:
        enemy["row"] += int(np.sign(5 - int(enemy["row"])))
        enemy["column"] += int(np.sign(5 - int(enemy["column"])))
      if self.boss is not None:
        self.boss["row"] += int(np.sign(5 - self.boss["row"]))
        self.boss["column"] += int(np.sign(5 - self.boss["column"]))
    if self.steps % max(5, 12 - self.era) == 0 and (self.enemies or self.boss is not None):
      source = self.enemies[0] if self.enemies else self.boss
      row = int(source["row"])
      column = int(source["column"])
      self.enemy_shots.append([row, column, int(np.sign(5 - row)), int(np.sign(5 - column))])

  def _move_enemy_shots(self) -> None:
    for shot in self.enemy_shots:
      if self.era >= 2:
        shot[2] = int(np.sign(5 - shot[0]))
        shot[3] = int(np.sign(5 - shot[1]))
      shot[0] += shot[2]
      shot[1] += shot[3]
    self.enemy_shots = [shot for shot in self.enemy_shots if 0 <= shot[0] < 10 and 0 <= shot[1] < 10]

  def _resolve_hits(self) -> float:
    reward = 0.0
    hit_bullets: set[int] = set()
    hit_enemy_ids: set[int] = set()
    for bullet_index, bullet in enumerate(self.bullets):
      for enemy in self.enemies:
        if bullet[:2] != [enemy["row"], enemy["column"]]:
          continue
        hit_bullets.add(bullet_index)
        hit_enemy_ids.add(int(enemy["id"]))
        if enemy["kind"] == "normal":
          reward += 100.0
          self.killed += 1
          if int(enemy["id"]) in FORMATION_IDS:
            self.formation_destroyed.add(int(enemy["id"]))
        else:
          reward += 1_500.0
          self.special_destroyed = True
        break
      if self.boss is not None and bullet[:2] == [self.boss["row"], self.boss["column"]]:
        hit_bullets.add(bullet_index)
        self.boss["hp"] -= 1
        if self.boss["hp"] == 0:
          reward += 3_000.0
          self.boss = None
          self.era += 1
          if self.era >= 5:
            self.terminal = True
          else:
            self._start_era()
          break
    self.enemies = [enemy for enemy in self.enemies if int(enemy["id"]) not in hit_enemy_ids]
    self.bullets = [bullet for index, bullet in enumerate(self.bullets) if index not in hit_bullets]
    if not self.formation_failed and self.formation_destroyed == FORMATION_IDS:
      reward += 2_000.0
      self.formation_failed = True
    return reward

  def _resolve_collisions(self) -> None:
    colliding_ids = {
      int(enemy["id"])
      for enemy in self.enemies
      if int(enemy["row"]) == 5 and int(enemy["column"]) == 5
    }
    collision = bool(colliding_ids) or any(shot[:2] == [5, 5] for shot in self.enemy_shots)
    if self.boss is not None and [self.boss["row"], self.boss["column"]] == [5, 5]:
      collision = True
    if not collision:
      return
    if colliding_ids.intersection(FORMATION_IDS):
      self.formation_failed = True
    for enemy in self.enemies:
      if int(enemy["id"]) in colliding_ids:
        enemy["row"] = 0
        enemy["column"] = 1 + int(enemy["id"]) % 8
    if self.boss is not None and [self.boss["row"], self.boss["column"]] == [5, 5]:
      self.boss["row"] = 0
      self.boss["column"] = 5
    self.bullets.clear()
    self.enemy_shots.clear()
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True

  def _spawn_boss_if_ready(self) -> None:
    if self.killed < 56 or self.enemies or not self.special_destroyed or self.boss is not None:
      return
    self.boss = {"row": 0, "column": 5, "hp": 7}

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    for enemy in self.enemies:
      channel = 2 if enemy["kind"] == "special" else 1
      state[int(enemy["row"]), int(enemy["column"]), channel] = True
    if self.boss is not None:
      state[self.boss["row"], self.boss["column"], 2] = True
    for bullet in self.bullets:
      state[bullet[0], bullet[1], 3] = True
    for shot in self.enemy_shots:
      state[shot[0], shot[1], 1] = True
    for row, column in self.previous_objects:
      if 0 <= row < 10 and 0 <= column < 10:
        state[row, column, 4] = True
    values = (
      (self.era, 3),
      (self.heading, 3),
      (self.previous_heading, 3),
      (self.lives, 5),
      (self.spawned, 6),
      (self.killed, 6),
      (self.steps, 13),
      (0 if self.boss is None else self.boss["hp"], 3),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
      offset += width
    for enemy_id in self.formation_destroyed:
      state[5, enemy_id - 20, 5] = True
    state[5, 5, 5] = self.formation_failed
    state[5, 6, 5] = self.special_spawned
    state[5, 7, 5] = self.special_destroyed
    return state
