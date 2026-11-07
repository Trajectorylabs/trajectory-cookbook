import numpy as np


class AtlantisEnvironment:
  """Three fixed defense posts protect seven installations from descending formations."""

  score_cap = 35_500.0

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
    self.installations = [True] * 7
    self.shots: list[list[int | str]] = []
    self.wave = 0
    self.wave_step = 0
    self.steps = 0
    self.total_score = 0.0
    self.restorations_used = 0
    self.last_action = 0
    self.terminal = False
    self.escaped = False
    self._start_wave()

  def _start_wave(self) -> None:
    self.enemies = [
      {"row": index // 5, "column": (index * 2) % 10, "direction": 1, "bandit": index == 8}
      for index in range(9)
    ]
    self.previous_enemies = [[int(enemy["row"]), int(enemy["column"])] for enemy in self.enemies]
    self.wave_step = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if action == 1:
      self.shots.append([8, 1, -1, 1, "sentry"])
    elif action == 3:
      self.shots.append([8, 8, -1, -1, "sentry"])
    elif action == 5 and self.installations[0]:
      self.shots.append([8, 5, -1, 0, "command"])

    self.previous_enemies = [[int(enemy["row"]), int(enemy["column"])] for enemy in self.enemies]
    self._move_enemies()
    self._move_shots()
    reward = self._resolve_hits()
    self._resolve_deathrays()
    self.wave_step += 1
    self.steps += 1
    if not any(self.installations):
      self.escaped = True
      self.terminal = True
    elif not self.enemies:
      bonus = 500.0 * sum(self.installations)
      reward += bonus
      self.total_score += bonus
      self._restore_installations()
      self.wave += 1
      if self.wave == 5:
        self.terminal = True
      else:
        self._start_wave()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_enemies(self) -> None:
    if self.wave_step % max(1, 4 - min(self.wave, 2)) != 0:
      return
    for enemy in self.enemies:
      column = int(enemy["column"]) + int(enemy["direction"])
      if column < 0 or column > 9:
        enemy["direction"] = -int(enemy["direction"])
        enemy["row"] = int(enemy["row"]) + 1
        column = min(9, max(0, column))
      enemy["column"] = column

  def _move_shots(self) -> None:
    for shot in self.shots:
      shot[0] = int(shot[0]) + int(shot[2])
      shot[1] = int(shot[1]) + int(shot[3])
    self.shots = [shot for shot in self.shots if 0 <= int(shot[0]) < 10 and 0 <= int(shot[1]) < 10]

  def _resolve_hits(self) -> float:
    reward = 0.0
    remaining_shots = []
    for shot in self.shots:
      hit = next(
        (enemy for enemy in self.enemies if [enemy["row"], enemy["column"]] == shot[:2]),
        None,
      )
      if hit is None:
        remaining_shots.append(shot)
        continue
      sentry = shot[4] == "sentry"
      if bool(hit["bandit"]):
        reward += 2_000.0 if sentry else 1_000.0
        self.enemies.clear()
        remaining_shots = []
        break
      reward += 200.0 if sentry else 100.0
      self.enemies.remove(hit)
    self.shots = remaining_shots
    self.total_score += reward
    return reward

  def _resolve_deathrays(self) -> None:
    attackers = [enemy for enemy in self.enemies if int(enemy["row"]) >= 8]
    for enemy in attackers:
      live = [index for index, intact in enumerate(self.installations) if intact]
      if live:
        target = 0 if self.installations[0] else live[int(self.random.randint(len(live)))]
        self.installations[target] = False
      self.enemies.remove(enemy)

  def _restore_installations(self) -> None:
    available = int(self.total_score // 10_000) - self.restorations_used
    while available > 0 and not all(self.installations):
      target = next(index for index, intact in enumerate(self.installations) if not intact)
      self.installations[target] = True
      self.restorations_used += 1
      available -= 1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 0, 0] = True
    state[9, 9, 0] = True
    if self.installations[0]:
      state[9, 5, 0] = True
    for enemy in self.enemies:
      row, column = int(enemy["row"]), int(enemy["column"])
      state[row, column, 1] = True
      state[row, column, 2] = True
      if bool(enemy["bandit"]):
        state[row, column, 5] = True
    for shot in self.shots:
      state[int(shot[0]), int(shot[1]), 3] = True
    for row, column in self.previous_enemies:
      if 0 <= row < 10 and 0 <= column < 10:
        state[row, column, 4] = True
    for index, intact in enumerate(self.installations):
      state[9, index + 1, 5] = intact
    state[8, : self.wave + 1, 5] = True
    return state
