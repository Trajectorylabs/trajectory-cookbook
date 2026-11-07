import numpy as np


TREASURE_VALUES = (
  (20, 15, 25, 25, 20, 45),
  (40, 30, 25, 40, 20, 20),
  (60, 35, 30, 25, 30, 5),
  (55, 40, 25, 80, 20, 40, 35),
)
CREATURE_VALUES = (1, 2, 3)
START = (8, 1)
EXIT = (1, 8)
WARP_LEFT = (5, 0)
WARP_RIGHT = (5, 9)


def _walls(chamber: int) -> set[tuple[int, int]]:
  walls = {(0, column) for column in range(10)} | {(9, column) for column in range(10)}
  walls |= {(row, 0) for row in range(1, 9) if row != 5}
  walls |= {(row, 9) for row in range(1, 9) if row != 5}
  patterns = (
    {(2, column) for column in range(1, 7)} | {(6, column) for column in range(3, 9)},
    {(row, 3) for row in range(1, 7)} | {(row, 6) for row in range(3, 9)},
    {(2, column) for column in range(3, 9)} | {(7, column) for column in range(1, 7)},
    {(row, 2) for row in range(2, 8)} | {(row, 7) for row in range(2, 8)},
  )
  return walls | patterns[chamber]


class TutankhamEnvironment:
  """Finite Atari 2600 Tutankham campaign through all sixteen chambers."""

  score_cap = 5_396.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 6_000):
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
    self.stage = 0
    self.lives = 3
    self.flashes = 3
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.facing = 1
    self.terminal = False
    self._start_chamber()

  def _start_chamber(self) -> None:
    self.chamber = self.stage % 4
    self.level = self.stage // 4
    self.player = list(START)
    self.ammunition = 100
    self.has_key = False
    self.walls = _walls(self.chamber)
    open_cells = [
      (row, column)
      for row in range(1, 9)
      for column in range(1, 9)
      if (row, column) not in self.walls and (row, column) not in (START, EXIT)
    ]
    treasure_count = len(TREASURE_VALUES[self.chamber])
    treasure_cells = open_cells[2 : 2 + treasure_count]
    self.treasures = {
      cell: (index, value)
      for index, (cell, value) in enumerate(zip(treasure_cells, TREASURE_VALUES[self.chamber]))
    }
    self.key_position = treasure_cells[0]
    self.nests = (open_cells[-1], open_cells[-2], open_cells[-3])
    self.enemies: list[dict[str, int]] = []
    self.spawned = 0
    self.killed = 0
    self.bullets: list[list[int]] = []

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    previous_action = self.last_action
    self.last_action = action
    reward = 0.0
    if action in (1, 2, 3, 4):
      reward += self._move_player(action)
    elif action == 5:
      if previous_action == 5 and self.flashes > 0:
        self.flashes -= 1
        self._flash_guardians()
      elif self.ammunition > 0:
        self.ammunition -= 1
        self.bullets.append([self.player[0], self.player[1], self.facing])
    self.ammunition = max(0, self.ammunition - 1)
    self._move_bullets()
    reward += self._resolve_bullet_hits()
    self._spawn_guardian()
    self._move_guardians()
    self._resolve_guardian_collision()
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> float:
    deltas = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}
    row_delta, column_delta = deltas[action]
    if action == 1:
      self.facing = -1
    elif action == 3:
      self.facing = 1
    destination = (self.player[0] + row_delta, self.player[1] + column_delta)
    if destination == WARP_LEFT:
      destination = WARP_RIGHT
    elif destination == WARP_RIGHT:
      destination = WARP_LEFT
    if destination in self.walls:
      return 0.0
    if destination == EXIT:
      if not self.has_key:
        return 0.0
      reward = float(self.ammunition)
      self.stage += 1
      if self.stage == 16:
        self.terminal = True
      else:
        self._start_chamber()
      return reward
    self.player[:] = destination
    treasure = self.treasures.pop(destination, None)
    if treasure is None:
      return 0.0
    index, value = treasure
    if index == 0:
      self.has_key = True
    return float(value)

  def _spawn_guardian(self) -> None:
    if self.spawned >= 18 or len(self.enemies) >= 3 or self.steps % 4:
      return
    kind = self.spawned % 3
    row, column = self.nests[kind]
    self.enemies.append({"id": self.spawned, "kind": kind, "row": row, "column": column})
    self.spawned += 1

  def _move_guardians(self) -> None:
    interval = max(1, 4 - self.level)
    if self.steps % interval:
      return
    for enemy in self.enemies:
      row_delta = int(np.sign(self.player[0] - enemy["row"]))
      column_delta = int(np.sign(self.player[1] - enemy["column"]))
      candidates = ((enemy["row"] + row_delta, enemy["column"]), (enemy["row"], enemy["column"] + column_delta))
      for destination in candidates:
        if destination not in self.walls:
          enemy["row"], enemy["column"] = destination
          break

  def _move_bullets(self) -> None:
    for bullet in self.bullets:
      bullet[1] += bullet[2]
    self.bullets = [
      bullet for bullet in self.bullets if 0 <= bullet[1] < 10 and tuple(bullet[:2]) not in self.walls
    ]

  def _resolve_bullet_hits(self) -> float:
    hit_bullets: set[int] = set()
    hit_enemies: set[int] = set()
    reward = 0.0
    for bullet_index, bullet in enumerate(self.bullets):
      for enemy in self.enemies:
        if bullet[:2] == [enemy["row"], enemy["column"]]:
          hit_bullets.add(bullet_index)
          hit_enemies.add(enemy["id"])
          reward += CREATURE_VALUES[enemy["kind"]]
          self.killed += 1
          break
    self.bullets = [bullet for index, bullet in enumerate(self.bullets) if index not in hit_bullets]
    self.enemies = [enemy for enemy in self.enemies if enemy["id"] not in hit_enemies]
    return float(reward)

  def _flash_guardians(self) -> None:
    for enemy in self.enemies:
      enemy["row"], enemy["column"] = self.nests[enemy["kind"]]

  def _resolve_guardian_collision(self) -> None:
    if not any([enemy["row"], enemy["column"]] == self.player for enemy in self.enemies):
      return
    self.lives -= 1
    self.player[:] = START
    self.enemies.clear()
    self.bullets.clear()
    if self.lives == 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.player[0], self.player[1], 0] = True
    for enemy in self.enemies:
      state[enemy["row"], enemy["column"], 1] = True
    for position, (index, _value) in self.treasures.items():
      state[position[0], position[1], 2 if index == 0 else 3] = True
    state[EXIT[0], EXIT[1], 2] = True
    for row, column in self.walls:
      state[row, column, 4] = True
    for bullet in self.bullets:
      state[bullet[0], bullet[1], 5] = True
    values = ((self.stage, 4), (self.lives, 2), (self.flashes, 2), (self.ammunition, 7), (self.spawned, 5), (self.killed, 5))
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[8 - offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[6, 0, 5] = self.has_key
    state[6, 1, 5] = self.facing > 0
    return state
