import numpy as np


ENEMY_VALUES = {
  "flagship": 500,
  "raider": 400,
  "pirate": 320,
  "targeter": 320,
  "glider": 320,
  "mechnoid": 300,
  "blockader": 260,
  "cobra": 80,
  "guardian": 60,
  "distractor": 20,
}
SECTORS = {
  (0, 1): ("enemies", ("flagship",)),
  (0, 2): ("enemies", ("raider",)),
  (0, 3): ("federation", ()),
  (1, 0): ("enemies", ("pirate",)),
  (1, 1): ("enemies", ("targeter",)),
  (1, 2): ("planet", ("guardian",)),
  (1, 3): ("enemies", ("glider",)),
  (2, 0): ("enemies", ("mechnoid",)),
  (2, 1): ("corridor", ("guardian",)),
  (2, 2): ("enemies", ("blockader",)),
  (2, 3): ("enemies", ("cobra",)),
  (3, 0): ("enemies", ("distractor",)),
  (3, 1): ("federation", ()),
  (3, 2): ("enemies", ("flagship", "mechnoid", "pirate")),
  (3, 3): ("solaris", ()),
}


class SolarisEnvironment:
  """Scanner navigation and targeting combat on a finite route to Solaris."""

  score_cap = 19_760.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 2_000):
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
    self.mode = "scanner"
    self.ship_row = 0
    self.ship_column = 0
    self.cursor_row = 0
    self.cursor_column = 0
    self.warp_target = (0, 0)
    self.focus_offset = 0
    self.fuel = 100
    self.lives = 3
    self.targeting_damaged = False
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.cleared: set[tuple[int, int]] = set()
    self.defeated: set[tuple[int, int, int]] = set()
    self.red_rows: set[int] = set()
    self.crosshair_row = 5
    self.crosshair_column = 5
    self.enemies: list[dict[str, int | str]] = []
    self.combat_steps = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if self.mode == "scanner":
      reward = self._act_scanner(action)
    elif self.mode == "warp":
      reward = self._act_warp(action)
    else:
      reward = self._act_combat(action)
    self.score += reward
    self.steps += 1
    if self.steps >= 40 and (0, 2) not in self.cleared:
      self.red_rows.add(0)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _act_scanner(self, action: int) -> float:
    if action == 5:
      self.warp_target = (self.cursor_row, self.cursor_column)
      self.focus_offset = int(self.random.randint(-3, 4))
      self.mode = "warp"
      return 0.0
    if self.ship_row in self.red_rows:
      action = {1: 3, 2: 4, 3: 1, 4: 2}.get(action, action)
    if action == 1:
      self.cursor_column = max(0, self.cursor_column - 1)
    elif action == 2:
      self.cursor_row = max(0, self.cursor_row - 1)
    elif action == 3:
      self.cursor_column = min(3, self.cursor_column + 1)
    elif action == 4:
      self.cursor_row = min(3, self.cursor_row + 1)
    return 0.0

  def _act_warp(self, action: int) -> float:
    if action == 1:
      self.focus_offset = min(3, self.focus_offset + 1)
      return 0.0
    if action == 3:
      self.focus_offset = max(-3, self.focus_offset - 1)
      return 0.0
    if action != 5:
      return 0.0
    distance = abs(self.warp_target[0] - self.ship_row) + abs(self.warp_target[1] - self.ship_column)
    self.fuel -= 4 + distance * 3 + abs(self.focus_offset) * 2
    self.ship_row, self.ship_column = self.warp_target
    self.cursor_row, self.cursor_column = self.warp_target
    if self.fuel <= 0:
      self._lose_ship()
      return 0.0
    sector_kind, enemy_kinds = SECTORS.get(self.warp_target, ("empty", ()))
    if sector_kind == "solaris":
      self.terminal = True
      self.mode = "scanner"
    elif sector_kind == "federation":
      self.fuel = 100
      self.targeting_damaged = False
      self.mode = "scanner"
    elif self.warp_target in self.cleared or not enemy_kinds:
      self.mode = "scanner"
    else:
      self._start_combat(enemy_kinds)
    return 0.0

  def _start_combat(self, enemy_kinds: tuple[str, ...]) -> None:
    self.mode = "combat"
    self.crosshair_row = 5
    self.crosshair_column = 5
    self.combat_steps = 0
    self.enemies = []
    for index, kind in enumerate(enemy_kinds):
      identity = (self.ship_row, self.ship_column, index)
      if identity in self.defeated:
        continue
      code = list(ENEMY_VALUES).index(kind)
      self.enemies.append(
        {
          "kind": kind,
          "index": index,
          "row": 1 + (code * 3 + index * 2) % 8,
          "column": 1 + (code * 5 + index * 3) % 8,
        }
      )

  def _act_combat(self, action: int) -> float:
    if action == 1:
      self.crosshair_column = max(0, self.crosshair_column - 1)
    elif action == 2:
      self.crosshair_row = max(0, self.crosshair_row - 1)
    elif action == 3:
      self.crosshair_column = min(9, self.crosshair_column + 1)
    elif action == 4:
      self.crosshair_row = min(9, self.crosshair_row + 1)
    reward = 0.0
    if action == 5 and self.enemies:
      enemy = self.enemies[0]
      if self.crosshair_row == enemy["row"] and self.crosshair_column == enemy["column"]:
        reward += float(ENEMY_VALUES[str(enemy["kind"])])
        self.defeated.add((self.ship_row, self.ship_column, int(enemy["index"])))
        self.enemies.pop(0)
        if not self.enemies:
          reward += self._clear_sector()
          return reward
    self.combat_steps += 1
    self.fuel -= 1
    if self.combat_steps % 10 == 0:
      self._enemy_hit()
    if self.fuel <= 0:
      self._lose_ship()
    return reward

  def _clear_sector(self) -> float:
    position = (self.ship_row, self.ship_column)
    self.cleared.add(position)
    kind = SECTORS[position][0]
    reward = 0.0
    if kind in {"planet", "corridor"}:
      reward = 8_000.0
      if kind == "planet":
        self.lives = min(9, self.lives + 1)
    self.mode = "scanner"
    self.combat_steps = 0
    return reward

  def _enemy_hit(self) -> None:
    if self.enemies and self.enemies[0]["kind"] == "flagship":
      self._lose_ship()
      return
    if not self.targeting_damaged:
      self.targeting_damaged = True
    else:
      self.fuel -= 15

  def _lose_ship(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    self.fuel = 100
    self.targeting_damaged = False
    self.mode = "scanner"
    self.enemies = []

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    if self.mode in {"scanner", "warp"}:
      state[self.ship_row * 2 + 1, self.ship_column * 2 + 1, 0] = True
      state[self.cursor_row * 2 + 1, self.cursor_column * 2 + 1, 3] = True
      for (row, column), (kind, _) in SECTORS.items():
        screen_row = row * 2 + 1
        screen_column = column * 2 + 1
        channel = 2 if kind in {"federation", "solaris"} else 1
        state[screen_row, screen_column, channel] = True
        if (row, column) in self.cleared:
          state[screen_row, screen_column, 4] = True
      for row in self.red_rows:
        state[row * 2 : row * 2 + 2, :8, 1] = True
    else:
      state[self.crosshair_row, self.crosshair_column, 0] = True
      if self.enemies:
        enemy = self.enemies[0]
        state[int(enemy["row"]), int(enemy["column"]), 1] = True
    for index, position in enumerate(SECTORS):
      state[index // 10, index % 10, 5] = position in self.cleared
    for row, column, enemy_index in self.defeated:
      index = 72 + (row * 4 + column + enemy_index) % 28
      state[index // 10, index % 10, 5] = True
    values = (
      (self.ship_row, 2),
      (self.ship_column, 2),
      (self.cursor_row, 2),
      (self.cursor_column, 2),
      (self.fuel, 7),
      (self.lives, 4),
      (self.focus_offset + 3, 3),
      (("scanner", "warp", "combat").index(self.mode), 2),
      (self.crosshair_row, 4),
      (self.crosshair_column, 4),
      (self.combat_steps, 8),
      (self.steps, 11),
    )
    offset = 16
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
      offset += width
    state[7, 0, 5] = self.targeting_damaged
    return state
