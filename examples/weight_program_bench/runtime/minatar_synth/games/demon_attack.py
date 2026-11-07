import numpy as np


PRIMARY_VALUES = (10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35)
SPLIT_VALUES = (0, 0, 0, 0, 40, 40, 50, 50, 60, 60, 70, 70)
DIVING_VALUES = (0, 0, 0, 0, 80, 80, 100, 100, 120, 120, 140, 140)


class DemonAttackEnvironment:
  """Twelve-wave fixed shooter with splitting and diving demon states."""

  score_cap = 12_180.0

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
    self.player = [9, 4]
    self.previous_player = self.player.copy()
    self.bunkers = 3
    self.wave = 1
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.player_shots: list[list[int]] = []
    self.enemy_shots: list[list[int]] = []
    self.previous_enemies: list[list[int]] = []
    self._start_wave()

  def _start_wave(self) -> None:
    self.enemies = [
      {
        "position": [1 + index % 3, 1 + (index * 2) % 9],
        "state": "primary",
        "direction": 1 if index % 2 == 0 else -1,
        "age": 0,
        "shooter": True,
      }
      for index in range(6)
    ]
    self.wave_steps = 0
    self.wave_hit = False
    self.enemy_shots = []

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_enemies = [list(enemy["position"]) for enemy in self.enemies]
    if action == 1:
      self.player[1] = max(0, self.player[1] - 1)
    elif action == 3:
      self.player[1] = min(9, self.player[1] + 1)
    elif action == 5 and len(self.player_shots) < 3:
      self.player_shots.append([8, self.player[1]])
    self._move_world()
    reward = self._resolve_player_shots()
    self._resolve_player_hits()
    if not self.enemies:
      if not self.wave_hit:
        self.bunkers = min(6, self.bunkers + 1)
      self.wave += 1
      if self.wave > 12:
        self.terminal = True
      else:
        self._start_wave()
    self.wave_steps += 1
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_world(self) -> None:
    for shot in self.player_shots:
      shot[0] -= 1
    self.player_shots = [shot for shot in self.player_shots if shot[0] >= 0]
    for enemy in self.enemies:
      position = enemy["position"]
      enemy["age"] = int(enemy["age"]) + 1
      if enemy["state"] == "diving":
        position[0] += 1
        position[1] += int(np.sign(self.player[1] - position[1]))
        if position[0] > 9:
          position[0] = 1
      else:
        position[1] += int(enemy["direction"])
        if position[1] <= 0 or position[1] >= 9:
          enemy["direction"] = -int(enemy["direction"])
          position[1] = int(np.clip(position[1], 0, 9))
        if enemy["state"] == "split" and int(enemy["age"]) >= 8:
          enemy["state"] = "diving"
      if bool(enemy["shooter"]) and self.random.rand() < min(0.25, 0.015 + self.wave * 0.008):
        self.enemy_shots.append([position[0] + 1, position[1]])
    for shot in self.enemy_shots:
      shot[0] += 1
    self.enemy_shots = [shot for shot in self.enemy_shots if shot[0] <= 9]

  def _resolve_player_shots(self) -> float:
    reward = 0.0
    surviving_shots = []
    destroyed_ids: set[int] = set()
    spawned: list[dict[str, object]] = []
    for shot in self.player_shots:
      enemy = next(
        (
          candidate
          for candidate in self.enemies
          if id(candidate) not in destroyed_ids and candidate["position"] == shot
        ),
        None,
      )
      if enemy is None:
        surviving_shots.append(shot)
        continue
      destroyed_ids.add(id(enemy))
      state = str(enemy["state"])
      if state == "primary":
        reward += PRIMARY_VALUES[self.wave - 1]
        if self.wave >= 5:
          for direction in (-1, 1):
            spawned.append(
              {
                "position": list(enemy["position"]),
                "state": "split",
                "direction": direction,
                "age": 0,
                "shooter": direction == 1,
              }
            )
      elif state == "split":
        reward += SPLIT_VALUES[self.wave - 1]
      else:
        reward += DIVING_VALUES[self.wave - 1]
    self.player_shots = surviving_shots
    self.enemies = [enemy for enemy in self.enemies if id(enemy) not in destroyed_ids] + spawned
    return reward

  def _resolve_player_hits(self) -> None:
    enemy_contact = any(enemy["position"] == self.player for enemy in self.enemies)
    shot_contact = self.player in self.enemy_shots
    if not enemy_contact and not shot_contact:
      return
    self.wave_hit = True
    self.enemy_shots = [shot for shot in self.enemy_shots if shot != self.player]
    if self.bunkers > 0:
      self.bunkers -= 1
    else:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies:
      position = enemy["position"]
      state[*position, 1] = True
      state[*position, 2] = True
      if enemy["state"] != "primary":
        state[*position, 5] = True
    for shot in self.enemy_shots:
      state[*shot, 1] = True
    for shot in self.player_shots:
      state[*shot, 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_enemies:
      state[*position, 4] = True
    state[9, : self.bunkers, 5] = True
    state[8, : min(10, self.wave), 5] = True
    return state
