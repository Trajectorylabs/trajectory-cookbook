import numpy as np


class PhoenixEnvironment:
  """Three five-stage Phoenix cycles with shield and layered mothership."""

  score_cap = 30_720.0

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
    self.cycle = 0
    self.stage = 0
    self.lives = 5
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_stage()

  def _start_stage(self) -> None:
    self.player_column = 5
    self.previous_player_column = 5
    self.shield_timer = 0
    self.shield_cooldown = 0
    self.shot: list[int] | None = None
    self.previous_shot: list[int] | None = None
    self.bombs: list[list[int]] = []
    self.previous_bombs: list[list[int]] = []
    self.stage_steps = 0
    if self.stage < 2:
      self.enemies = [
        {"id": index, "position": [1 + index // 4, 1 + (index % 4) * 2], "direction": 1 if index % 2 == 0 else -1}
        for index in range(8)
      ]
      self.wing_rewarded: set[int] = set()
    elif self.stage < 4:
      self.enemies = [
        {"id": index, "position": [1 + index // 2, 2 + (index % 2) * 5], "direction": 1 if index % 2 == 0 else -1}
        for index in range(4)
      ]
      self.wing_rewarded = set()
    else:
      self.enemies = []
      self.hull_health = [0, 0, 2, 2, 2, 2, 2, 2, 0, 0]
      self.rotating_hole = 2
      self.rotating_direction = 1
      self.mothership_shots = 0
    self.previous_enemies = {int(enemy["id"]): list(enemy["position"]) for enemy in self.enemies}

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player_column = self.player_column
    self.previous_shot = None if self.shot is None else self.shot.copy()
    self.previous_bombs = [bomb.copy() for bomb in self.bombs]
    self.previous_enemies = {int(enemy["id"]): list(enemy["position"]) for enemy in self.enemies}
    if self.shield_timer == 0:
      if action == 1:
        self.player_column = max(0, self.player_column - 1)
      elif action == 3:
        self.player_column = min(9, self.player_column + 1)
      elif action == 4 and self.shield_cooldown == 0:
        self.shield_timer = 3
        self.shield_cooldown = 10
    if action == 5 and self.shot is None:
      self.shot = [7, self.player_column]
    stage_before = (self.cycle, self.stage)
    reward = self._move_shot()
    if (self.cycle, self.stage) == stage_before and not self.terminal:
      if self.stage < 4:
        self._move_birds()
      else:
        self._move_mothership()
      self._move_bombs()
      reward += self._resolve_collisions()
    self.score += reward
    self.steps += 1
    self.stage_steps += 1
    self.shield_timer = max(0, self.shield_timer - 1)
    self.shield_cooldown = max(0, self.shield_cooldown - 1)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_shot(self) -> float:
    if self.shot is None:
      return 0.0
    self.shot[0] -= 1
    if self.stage < 2:
      for enemy in list(self.enemies):
        if self.shot == enemy["position"]:
          reward = 80.0 if enemy["position"][0] >= 5 else 20.0
          self.enemies.remove(enemy)
          self.shot = None
          self._advance_if_clear()
          return reward
    elif self.stage < 4:
      for enemy in list(self.enemies):
        row, column = enemy["position"]
        if self.shot[0] != row:
          continue
        if self.shot[1] == column:
          reward = float(100 + min(4, row - 1) * 100)
          self.enemies.remove(enemy)
          self.shot = None
          self._advance_if_clear()
          return reward
        if abs(self.shot[1] - column) == 1:
          reward = 0.0
          if enemy["id"] not in self.wing_rewarded:
            self.wing_rewarded.add(enemy["id"])
            reward = 20.0
          self.shot = None
          return reward
    else:
      self.mothership_shots += 1
      row, column = self.shot
      if row == 2 and self.hull_health[column] > 0:
        self.hull_health[column] -= 1
        self.shot = None
        return 0.0
      if row == 1 and column == 5 and self.hull_health[5] == 0 and self.rotating_hole == 5:
        maximum = 4_000 + self.cycle * 1_000
        reward = float(max(1_000, maximum - max(0, self.mothership_shots - 1) * 100))
        self.shot = None
        self._advance_stage()
        return reward
    if self.shot is not None and self.shot[0] < 0:
      self.shot = None
    return 0.0

  def _move_birds(self) -> None:
    for enemy in self.enemies:
      row, column = enemy["position"]
      if self.stage_steps % max(1, 3 - self.cycle) == 0:
        column += enemy["direction"]
        if column <= 0 or column >= 9:
          enemy["direction"] *= -1
          column = min(9, max(0, column))
          row = min(7, row + 1)
      enemy["position"] = [row, column]
      if (self.stage_steps + enemy["id"]) % max(5, 12 - self.cycle * 2) == 0:
        self.bombs.append([row + 1, column])

  def _move_mothership(self) -> None:
    if self.stage_steps % 2 == 0:
      self.rotating_hole += self.rotating_direction
      if self.rotating_hole in (2, 7):
        self.rotating_direction *= -1
    interval = max(3, 8 - self.cycle * 2)
    if self.stage_steps % interval == 0:
      self.bombs.append([2, int(self.random.randint(2, 8))])

  def _move_bombs(self) -> None:
    for bomb in list(self.bombs):
      bomb[0] += 1
      if bomb[0] > 8:
        self.bombs.remove(bomb)

  def _resolve_collisions(self) -> float:
    reward = 0.0
    for enemy in list(self.enemies):
      if enemy["position"] != [8, self.player_column]:
        continue
      if self.shield_timer:
        reward += self._shield_bird_value(enemy)
        self.enemies.remove(enemy)
        self._advance_if_clear()
      else:
        self._lose_life()
      break
    for bomb in list(self.bombs):
      if bomb != [8, self.player_column]:
        continue
      self.bombs.remove(bomb)
      if not self.shield_timer:
        self._lose_life()
      break
    return reward

  def _shield_bird_value(self, enemy: dict[str, object]) -> float:
    if self.stage < 2:
      return 80.0
    return 500.0

  def _advance_if_clear(self) -> None:
    if not self.enemies:
      self._advance_stage()

  def _advance_stage(self) -> None:
    self.stage += 1
    if self.stage >= 5:
      self.stage = 0
      self.cycle += 1
    if self.cycle >= 3:
      self.terminal = True
    else:
      self._start_stage()

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self.player_column = 5
      self.shield_timer = 0
      self.shield_cooldown = 0
      self.shot = None
      self.bombs.clear()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, self.player_column, 0] = True
    state[8, self.previous_player_column, 4] = True
    for enemy in self.enemies:
      row, column = enemy["position"]
      state[row, column, 1] = True
      state[*self.previous_enemies[int(enemy["id"])], 4] = True
      if self.stage >= 2:
        state[row, max(0, column - 1) : min(10, column + 2), 1] = True
        if enemy["id"] in self.wing_rewarded:
          state[row, column, 3] = True
    for bomb in self.bombs:
      state[*bomb, 1] = True
    for bomb in self.previous_bombs:
      state[*bomb, 4] = True
    if self.shot is not None:
      state[*self.shot, 2] = True
    if self.previous_shot is not None:
      state[*self.previous_shot, 4] = True
    if self.stage == 4:
      for column in range(2, 8):
        if self.hull_health[column]:
          state[2, column, 3] = True
      state[1, 5, 1] = True
      state[3, self.rotating_hole, 3] = True
    if self.shield_timer:
      state[7:10, max(0, self.player_column - 1) : min(10, self.player_column + 2), 3] = True
    state[9, : self.lives, 2] = True
    for bit in range(2):
      state[0, bit, 5] = bool(self.cycle & (1 << bit))
    for bit in range(3):
      state[0, 2 + bit, 5] = bool(self.stage & (1 << bit))
      state[0, 5 + bit, 5] = bool(self.lives & (1 << bit))
    for bit in range(4):
      state[1, bit, 5] = bool(self.shield_cooldown & (1 << bit))
      state[1, 4 + bit, 5] = bool(self.shield_timer & (1 << bit))
    for index in self.wing_rewarded:
      state[2, index, 5] = True
    for bit in range(12):
      state[3 + bit // 10, bit % 10, 5] = bool(self.steps & (1 << bit))
      state[5 + bit // 10, bit % 10, 5] = bool(self.stage_steps & (1 << bit))
    if self.stage == 4:
      for column in range(2, 8):
        for bit in range(2):
          state[7 + bit, column, 5] = bool(self.hull_health[column] & (1 << bit))
      state[9, self.rotating_hole, 5] = True
      state[9, 8, 5] = self.rotating_direction > 0
    return state
