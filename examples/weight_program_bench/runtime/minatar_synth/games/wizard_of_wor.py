import numpy as np


DIRECTIONS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}
SCORES = {"burwor": 100, "garwor": 200, "thorwor": 500, "worluk": 1_000, "wizard": 2_500}


def maze_walls(dungeon: int) -> set[tuple[int, int]]:
  walls = {(0, column) for column in range(10)} | {(9, column) for column in range(10)}
  walls |= {(row, 0) for row in range(1, 9) if row != 5}
  walls |= {(row, 9) for row in range(1, 9) if row != 5}
  patterns = (
    {(2, column) for column in (2, 3, 6, 7)} | {(6, column) for column in (2, 3, 6, 7)},
    {(row, 3) for row in (2, 3, 6, 7)} | {(row, 6) for row in (2, 3, 6, 7)},
    {(3, column) for column in (1, 2, 7, 8)} | {(6, column) for column in (1, 2, 7, 8)},
    {(row, 2) for row in (2, 3, 6, 7)} | {(row, 7) for row in (2, 3, 6, 7)},
  )
  return walls | patterns[dungeon]


class WizardOfWorEnvironment:
  """Four finite dungeons with radar, Worluk, and Wizard battles."""

  score_cap = 27_000.0

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
    self.dungeon = 0
    self.lives = 3
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.facing = 2
    self.double_score = False
    self.extra_lives_awarded = 0
    self.terminal = False
    self._start_dungeon()

  def _start_dungeon(self) -> None:
    self.player = [8, 5]
    self.walls = maze_walls(self.dungeon)
    self.phase = "worlings"
    self.enemies = [self._enemy(index, "burwor") for index in range(6)]
    self.bullet: list[int] | None = None
    self.enemy_shots: list[list[int]] = []
    self.worluk_killed = False

  def _enemy(self, identifier: int, kind: str) -> dict[str, int | str]:
    cells = ((1, 1), (1, 4), (1, 8), (4, 1), (4, 8), (7, 8))
    row, column = cells[identifier % len(cells)]
    return {"id": identifier, "kind": kind, "row": row, "column": column}

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if action in DIRECTIONS:
      self.facing = action
      self._move_player(action)
    elif action == 5 and self.bullet is None:
      row_delta, column_delta = DIRECTIONS[self.facing]
      self.bullet = [self.player[0], self.player[1], row_delta, column_delta]
    self._move_bullet()
    reward = self._resolve_hits()
    self._move_enemies()
    self._move_enemy_shots()
    self._resolve_player_collision()
    self._advance_phase()
    self.score += reward
    while self.score >= 10_000 * (1 + getattr(self, "extra_lives_awarded", 0)):
      self.lives += 1
      self.extra_lives_awarded = getattr(self, "extra_lives_awarded", 0) + 1
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    row_delta, column_delta = DIRECTIONS[action]
    destination = [self.player[0] + row_delta, self.player[1] + column_delta]
    if destination == [5, 0] and self._doors_open():
      destination = [5, 8]
    elif destination == [5, 9] and self._doors_open():
      destination = [5, 1]
    if tuple(destination) not in self.walls:
      self.player = destination

  def _doors_open(self) -> bool:
    return self.steps % 6 < 3

  def _move_bullet(self) -> None:
    if self.bullet is None:
      return
    self.bullet[0] += self.bullet[2]
    self.bullet[1] += self.bullet[3]
    if tuple(self.bullet[:2]) in self.walls or not (0 <= self.bullet[0] < 10 and 0 <= self.bullet[1] < 10):
      self.bullet = None

  def _resolve_hits(self) -> float:
    if self.bullet is None:
      return 0.0
    for enemy in self.enemies:
      if self.bullet[:2] != [enemy["row"], enemy["column"]]:
        continue
      self.enemies.remove(enemy)
      self.bullet = None
      kind = str(enemy["kind"])
      multiplier = 2 if self.double_score and kind != "wizard" else 1
      if kind == "worluk":
        self.worluk_killed = True
      return float(SCORES[kind] * multiplier)
    return 0.0

  def _move_enemies(self) -> None:
    interval = max(1, 4 - self.dungeon)
    if self.steps % interval:
      return
    for enemy in self.enemies:
      kind = str(enemy["kind"])
      if kind == "wizard" and self.steps % 4 == 0:
        cells = [(row, column) for row in range(1, 9) for column in range(1, 9) if (row, column) not in self.walls]
        enemy["row"], enemy["column"] = cells[int(self.random.randint(len(cells)))]
      elif kind == "worluk":
        enemy["column"] += 1
      else:
        enemy["row"] += int(np.sign(self.player[0] - int(enemy["row"])))
        enemy["column"] += int(np.sign(self.player[1] - int(enemy["column"])))
      if kind in ("garwor", "thorwor", "wizard") and self.steps % 7 == 0:
        self.enemy_shots.append([
          int(enemy["row"]), int(enemy["column"]),
          int(np.sign(self.player[0] - int(enemy["row"]))),
          int(np.sign(self.player[1] - int(enemy["column"]))),
        ])

  def _move_enemy_shots(self) -> None:
    for shot in self.enemy_shots:
      shot[0] += shot[2]
      shot[1] += shot[3]
    self.enemy_shots = [shot for shot in self.enemy_shots if 0 <= shot[0] < 10 and 0 <= shot[1] < 10 and tuple(shot[:2]) not in self.walls]

  def _resolve_player_collision(self) -> None:
    collision = any([enemy["row"], enemy["column"]] == self.player for enemy in self.enemies)
    collision |= any(shot[:2] == self.player for shot in self.enemy_shots)
    if not collision:
      return
    self.lives -= 1
    self.player = [8, 5]
    self.bullet = None
    self.enemy_shots.clear()
    if self.lives <= 0:
      self.terminal = True

  def _advance_phase(self) -> None:
    self.enemies = [enemy for enemy in self.enemies if not (enemy["kind"] == "worluk" and int(enemy["column"]) >= 9)]
    if self.enemies or self.terminal:
      return
    if self.phase == "worlings":
      self.phase = "garwors"
      self.enemies = [self._enemy(index, "garwor") for index in range(2)]
    elif self.phase == "garwors":
      self.phase = "thorwors"
      self.enemies = [self._enemy(index, "thorwor") for index in range(2)]
    elif self.phase == "thorwors" and self.dungeon > 0:
      self.phase = "worluk"
      self.enemies = [self._enemy(0, "worluk")]
    elif self.phase in ("thorwors", "worluk"):
      self.phase = "wizard"
      self.enemies = [self._enemy(0, "wizard")]
    else:
      next_double = self.worluk_killed
      self.dungeon += 1
      if self.dungeon == 4:
        self.terminal = True
      else:
        self.double_score = next_double
        self._start_dungeon()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.player[0], self.player[1], 0] = True
    for enemy in self.enemies:
      visible = enemy["kind"] not in ("garwor", "worluk") or self.steps % 4 == 0
      if visible:
        state[int(enemy["row"]), int(enemy["column"]), 1] = True
      state[int(enemy["row"]), int(enemy["column"]), 2] = True
    for shot in self.enemy_shots:
      state[shot[0], shot[1], 1] = True
    if self.bullet is not None:
      state[self.bullet[0], self.bullet[1], 3] = True
    for row, column in self.walls:
      state[row, column, 4] = True
    values = ((self.dungeon, 3), (self.lives, 4), (len(self.enemies), 4), (self.steps, 12))
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[3, 0, 5] = self.double_score
    state[3, 1, 5] = self._doors_open()
    return state
