import numpy as np


TENTACLE_COLUMNS = (2, 4, 6, 8)
SHARK_VALUES_BY_ROW = (0, 10, 20, 30, 40, 50, 80, 100, 100)


class NameThisGameEnvironment:
  """Eight escalating treasure-defense waves under a moving oxygen line."""

  score_cap = 4_800.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 3_000):
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
    self.wave = 0
    self.treasure = 5
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_wave()

  def _start_wave(self) -> None:
    self.diver_column = 5
    self.previous_diver_column = self.diver_column
    self.air = 100
    self.air_line_column = 0
    self.air_line_direction = 1
    self.air_bonus_units = 0
    self.tentacle_lengths = [1, 1, 1, 1]
    self.tentacles_destroyed: set[int] = set()
    self.shark_index = 0
    self.shark = [1, 0]
    self.previous_shark = self.shark.copy()
    self.shark_direction = 1
    self.projectiles: list[list[int]] = []
    self.previous_projectiles: list[list[int]] = []
    self.wave_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_diver_column = self.diver_column
    self.previous_shark = self.shark.copy()
    self.previous_projectiles = [projectile.copy() for projectile in self.projectiles]
    if action == 1:
      self.diver_column = max(0, self.diver_column - 1)
    elif action == 3:
      self.diver_column = min(9, self.diver_column + 1)
    elif action == 5 and len(self.projectiles) < 3:
      self.projectiles.append([7, self.diver_column])
    wave_before = self.wave
    reward = self._move_projectiles()
    if self.wave == wave_before and not self.terminal:
      self._move_shark()
      self._grow_tentacles()
      reward += self._move_air_line()
      self.air -= 1 + self.wave // 3
      self._resolve_failures()
    self.score += reward
    self.steps += 1
    self.wave_steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_projectiles(self) -> float:
    reward = 0.0
    for projectile in list(self.projectiles):
      projectile[0] -= 1
      hit_tentacle = None
      for index, column in enumerate(TENTACLE_COLUMNS):
        if index in self.tentacles_destroyed:
          continue
        tip_row = self.tentacle_lengths[index]
        if projectile == [tip_row, column]:
          hit_tentacle = index
          break
      if hit_tentacle is not None:
        self.tentacles_destroyed.add(hit_tentacle)
        self.projectiles.remove(projectile)
        reward += 50.0
        continue
      if projectile == self.shark and self.shark_index < 3:
        reward += float(SHARK_VALUES_BY_ROW[self.shark[0]])
        self.shark_index += 1
        self.projectiles.remove(projectile)
        self._reset_shark()
        continue
      if projectile[0] < 0:
        self.projectiles.remove(projectile)
    if len(self.tentacles_destroyed) == 4 and self.shark_index >= 3:
      self.wave += 1
      if self.wave >= 8:
        self.terminal = True
      else:
        self._start_wave()
    return reward

  def _reset_shark(self) -> None:
    self.shark = [1, 0 if self.shark_direction > 0 else 9]
    self.previous_shark = self.shark.copy()

  def _move_shark(self) -> None:
    if self.shark_index >= 3:
      return
    interval = max(1, 3 - self.wave // 3)
    if self.wave_steps % interval:
      return
    self.shark[1] += self.shark_direction
    if self.shark[1] in (0, 9):
      self.shark_direction *= -1
      self.shark[0] = min(8, self.shark[0] + 1)

  def _grow_tentacles(self) -> None:
    interval = max(2, 8 - self.wave)
    if self.wave_steps % interval:
      return
    live = [index for index in range(4) if index not in self.tentacles_destroyed]
    if live:
      index = live[(self.wave_steps // interval) % len(live)]
      self.tentacle_lengths[index] = min(8, self.tentacle_lengths[index] + 1)

  def _move_air_line(self) -> float:
    self.air_line_column += self.air_line_direction
    if self.air_line_column in (0, 9):
      self.air_line_direction *= -1
    if self.air_line_column != self.diver_column:
      return 0.0
    if self.air < 100:
      self.air = min(100, self.air + 20)
      return 0.0
    if self.air_bonus_units < 10:
      self.air_bonus_units += 1
      return 10.0
    return 0.0

  def _resolve_failures(self) -> None:
    shark_hit = self.shark_index < 3 and self.shark == [8, self.diver_column]
    tentacle_hit = any(
      index not in self.tentacles_destroyed and length >= 8
      for index, length in enumerate(self.tentacle_lengths)
    )
    if self.air <= 0 or shark_hit or tentacle_hit:
      self.treasure -= 1
      if self.treasure <= 0:
        self.terminal = True
        return
      self.diver_column = 5
      self.air = 100
      self._reset_shark()
      for index in range(4):
        if index not in self.tentacles_destroyed:
          self.tentacle_lengths[index] = 1
      self.projectiles.clear()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, self.diver_column, 0] = True
    state[8, self.previous_diver_column, 4] = True
    if self.shark_index < 3:
      state[*self.shark, 1] = True
      state[*self.previous_shark, 4] = True
    for index, column in enumerate(TENTACLE_COLUMNS):
      if index in self.tentacles_destroyed:
        continue
      state[0 : self.tentacle_lengths[index] + 1, column, 1] = True
    for projectile in self.projectiles:
      state[*projectile, 2] = True
    for projectile in self.previous_projectiles:
      state[*projectile, 4] = True
    state[0:8, self.air_line_column, 3] = True
    state[9, : self.treasure, 2] = True
    for bit in range(3):
      state[0, bit, 5] = bool(self.wave & (1 << bit))
      state[0, 3 + bit, 5] = bool(self.treasure & (1 << bit))
      state[0, 6 + bit, 5] = bool(self.shark_index & (1 << bit))
    for bit in range(7):
      state[1, bit, 5] = bool(max(0, self.air) & (1 << bit))
    for bit in range(4):
      state[2, bit, 5] = bool(self.air_bonus_units & (1 << bit))
    for index in range(4):
      state[2, 4 + index, 5] = index in self.tentacles_destroyed
    for bit in range(12):
      state[3 + bit // 10, bit % 10, 5] = bool(self.steps & (1 << bit))
      state[5 + bit // 10, bit % 10, 5] = bool(self.wave_steps & (1 << bit))
    state[7, 0, 5] = self.shark_direction > 0
    state[7, 1, 5] = self.air_line_direction > 0
    return state
