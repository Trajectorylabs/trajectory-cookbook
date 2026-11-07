import numpy as np


class AssaultEnvironment:
  """Fixed-cannon wave defense with vertical/lateral fire and weapon heat."""

  score_cap = 75.0

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
    self.player_column = 5
    self.facing = 1
    self.mothership_column = 0
    self.mothership_direction = 1
    self.aliens: list[dict[str, int | bool]] = []
    self.friendly_shots: list[list[int]] = []
    self.hostile_shots: list[list[int]] = []
    self.fireballs: list[list[int]] = []
    self.previous_aliens: list[list[int]] = []
    self.previous_hostiles: list[list[int]] = []
    self.wave = 0
    self.spawned = 0
    self.lives = 3
    self.heat = 0
    self.steps = 0
    self.last_action = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    fired = False
    if action == 1:
      self.player_column = max(0, self.player_column - 1)
      self.facing = -1
    elif action == 3:
      self.player_column = min(9, self.player_column + 1)
      self.facing = 1
    elif action == 2 and self.heat < 9:
      self.friendly_shots.append([8, self.player_column, -1, 0])
      fired = True
    elif action == 5 and self.heat < 9:
      self.friendly_shots.append([9, self.player_column, 0, self.facing])
      fired = True
    self.heat = min(10, self.heat + 2) if fired else max(0, self.heat - 1)
    if self.heat >= 10:
      self._lose_life()

    self._move_mothership()
    self._spawn_alien()
    self.previous_aliens = [[int(alien["row"]), int(alien["column"])] for alien in self.aliens]
    self.previous_hostiles = [shot[:2] for shot in self.hostile_shots + self.fireballs]
    self._move_aliens()
    self._move_projectiles()
    reward = self._resolve_friendly_hits()
    self._resolve_hostile_hits()
    self._launch_hostile_weapon()
    self.steps += 1
    if self.spawned == 10 and not self.aliens:
      self.wave += 1
      if self.wave == 5:
        self.terminal = True
      else:
        self.spawned = 0
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_mothership(self) -> None:
    self.mothership_column += self.mothership_direction
    if self.mothership_column in (0, 9):
      self.mothership_direction *= -1

  def _spawn_alien(self) -> None:
    if self.spawned >= 10 or self.steps % 4 != 0:
      return
    self.aliens.append({
      "row": 1,
      "column": self.mothership_column,
      "direction": self.mothership_direction,
      "split": self.spawned % 2 == 1,
    })
    self.spawned += 1

  def _move_aliens(self) -> None:
    for alien in self.aliens:
      direction = int(alien["direction"])
      column = int(alien["column"]) + direction
      if column < 0 or column > 9:
        direction *= -1
        column = min(9, max(0, column))
      alien["column"] = column
      alien["direction"] = direction
      if self.wave % 3 == 1 and self.steps % 3 == 0:
        alien["row"] = min(7, int(alien["row"]) + 1)
      elif self.wave % 3 == 2 and self.random.rand() < 0.2:
        alien["row"] = int(self.random.randint(1, 7))
        alien["column"] = int(self.random.randint(10))

  def _move_projectiles(self) -> None:
    for shot in self.friendly_shots:
      shot[0] += shot[2]
      shot[1] += shot[3]
    self.friendly_shots = [shot for shot in self.friendly_shots if 0 <= shot[0] < 10 and 0 <= shot[1] < 10]
    for shot in self.hostile_shots:
      shot[0] += 1
    reached_base = [shot for shot in self.hostile_shots if shot[0] >= 9]
    self.hostile_shots = [shot for shot in self.hostile_shots if shot[0] < 9]
    for shot in reached_base:
      direction = int(np.sign(self.player_column - shot[1])) or 1
      self.fireballs.append([9, shot[1], 0, direction])
    for fireball in self.fireballs:
      fireball[1] += fireball[3]
    self.fireballs = [shot for shot in self.fireballs if 0 <= shot[1] < 10]

  def _resolve_friendly_hits(self) -> float:
    reward = 0.0
    remaining_shots = []
    for shot in self.friendly_shots:
      hit_fireball = next((fireball for fireball in self.fireballs if fireball[:2] == shot[:2]), None)
      if hit_fireball is not None:
        self.fireballs.remove(hit_fireball)
        continue
      hit = next((alien for alien in self.aliens if [alien["row"], alien["column"]] == shot[:2]), None)
      if hit is None:
        remaining_shots.append(shot)
        continue
      self.aliens.remove(hit)
      if bool(hit["split"]):
        column = int(hit["column"])
        row = int(hit["row"])
        for direction in (-1, 1):
          self.aliens.append({"row": row, "column": column, "direction": direction, "split": False})
      else:
        reward += 1.0
    self.friendly_shots = remaining_shots
    return reward

  def _resolve_hostile_hits(self) -> None:
    if any(shot[:2] == [9, self.player_column] for shot in self.fireballs):
      self.fireballs = [shot for shot in self.fireballs if shot[:2] != [9, self.player_column]]
      self._lose_life()

  def _launch_hostile_weapon(self) -> None:
    if self.steps % 12 != 0 or not self.aliens:
      return
    lowest = max(self.aliens, key=lambda alien: int(alien["row"]))
    self.hostile_shots.append([int(lowest["row"]) + 1, int(lowest["column"])])

  def _lose_life(self) -> None:
    self.lives -= 1
    self.heat = 0
    self.friendly_shots.clear()
    self.hostile_shots.clear()
    self.fireballs.clear()
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, self.player_column, 0] = True
    state[0, self.mothership_column, 1] = True
    for shot in self.hostile_shots + self.fireballs:
      state[shot[0], shot[1], 1] = True
    for alien in self.aliens:
      state[int(alien["row"]), int(alien["column"]), 2] = True
    for shot in self.friendly_shots:
      state[shot[0], shot[1], 3] = True
    for row, column in self.previous_aliens + self.previous_hostiles:
      state[row, column, 4] = True
    state[9, : self.heat, 5] = True
    state[8, : self.lives, 5] = True
    state[7, : self.wave + 1, 5] = True
    return state
