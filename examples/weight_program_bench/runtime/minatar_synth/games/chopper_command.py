import numpy as np


class ChopperCommandEnvironment:
  """Ten-wave convoy-defense shooter over a wraparound scanner world."""

  score_cap = 90_000.0

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
    self.player = [4, 1]
    self.previous_player = self.player.copy()
    self.facing = 1
    self.shots: list[dict[str, object]] = []
    self.missiles: list[dict[str, object]] = []
    self.trucks = [[8, column % 10] for column in range(12)]
    self.lives = 3
    self.score = 0.0
    self.next_life_score = 10_000.0
    self.wave = 1
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.previous_threats: list[list[int]] = []
    self._start_wave()

  def _start_wave(self) -> None:
    self.enemies = [
      {
        "position": [1 + index % 4, (index * 3 + self.wave) % 10],
        "kind": "jet" if index % 2 else "helicopter",
      }
      for index in range(12)
    ]
    self.wave_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_threats = self._threat_positions()
    self._apply_action(action)
    self._move_world()
    reward = self._resolve_shots()
    reward += self._resolve_missiles_and_contacts()
    self.score += reward
    if not self.enemies:
      reward += len(self.trucks) * self.wave * 100.0
      self.score += len(self.trucks) * self.wave * 100.0
      self.wave += 1
      if self.wave > 10:
        self.terminal = True
      else:
        self._start_wave()
    while self.score >= self.next_life_score and self.lives < 6:
      self.lives += 1
      self.next_life_score += 10_000.0
    self.wave_steps += 1
    self.steps += 1
    if self.steps >= self.horizon or not self.trucks:
      self.terminal = True
    return reward, self.terminal

  def _apply_action(self, action: int) -> None:
    if action == 1:
      self.player[1] = (self.player[1] - 1) % 10
      self.facing = -1
    elif action == 2:
      self.player[0] = max(0, self.player[0] - 1)
    elif action == 3:
      self.player[1] = (self.player[1] + 1) % 10
      self.facing = 1
    elif action == 4:
      self.player[0] = min(6, self.player[0] + 1)
    elif action == 5 and len(self.shots) < 3:
      self.shots.append({"position": self.player.copy(), "direction": self.facing})

  def _move_world(self) -> None:
    for shot in self.shots:
      shot["position"][1] = (shot["position"][1] + int(shot["direction"])) % 10
      shot["distance"] = int(shot.get("distance", 0)) + 1
    self.shots = [shot for shot in self.shots if int(shot["distance"]) < 10]
    move_interval = max(1, 5 - (self.wave - 1) // 2)
    if self.wave_steps % move_interval == 0:
      for enemy in self.enemies:
        enemy["position"][1] = (enemy["position"][1] - 1) % 10
    fire_probability = min(0.45, 0.04 + 0.025 * self.wave)
    for enemy in self.enemies:
      if self.random.rand() < fire_probability / max(1, len(self.enemies)):
        self.missiles.append({"position": list(enemy["position"]), "split": False, "direction": 0})
    new_missiles = []
    for missile in self.missiles:
      missile["position"][0] += 1
      missile["position"][1] = (missile["position"][1] + int(missile["direction"])) % 10
      if missile["position"][0] == 4 and not bool(missile["split"]):
        for direction in (-1, 1):
          new_missiles.append({"position": list(missile["position"]), "split": True, "direction": direction})
      elif missile["position"][0] <= 8:
        new_missiles.append(missile)
    self.missiles = new_missiles
    if self.wave_steps % 2 == 0:
      for truck in self.trucks:
        truck[1] = (truck[1] - 1) % 10

  def _resolve_shots(self) -> float:
    reward = 0.0
    remaining_shots = []
    hit_enemy_ids: set[int] = set()
    for shot in self.shots:
      hit = next(
        (
          enemy
          for enemy in self.enemies
          if id(enemy) not in hit_enemy_ids and enemy["position"] == shot["position"]
        ),
        None,
      )
      if hit is None:
        remaining_shots.append(shot)
        continue
      hit_enemy_ids.add(id(hit))
      reward += 200.0 if hit["kind"] == "jet" else 100.0
    self.shots = remaining_shots
    self.enemies = [enemy for enemy in self.enemies if id(enemy) not in hit_enemy_ids]
    return reward

  def _resolve_missiles_and_contacts(self) -> float:
    hit_truck_ids: set[int] = set()
    remaining_missiles = []
    player_hit = any(enemy["position"] == self.player for enemy in self.enemies)
    for missile in self.missiles:
      if missile["position"] == self.player:
        player_hit = True
        continue
      truck = next(
        (
          truck
          for truck in self.trucks
          if id(truck) not in hit_truck_ids and truck == missile["position"]
        ),
        None,
      )
      if truck is not None:
        hit_truck_ids.add(id(truck))
      else:
        remaining_missiles.append(missile)
    self.trucks = [truck for truck in self.trucks if id(truck) not in hit_truck_ids]
    self.missiles = remaining_missiles
    if player_hit:
      self.lives -= 1
      self.player = [4, 1]
      self.previous_player = self.player.copy()
      self.shots = []
      self.missiles = []
      if self.lives <= 0:
        self.terminal = True
    return 0.0

  def _threat_positions(self) -> list[list[int]]:
    positions = [list(enemy["position"]) for enemy in self.enemies]
    positions.extend(list(missile["position"]) for missile in self.missiles)
    return positions

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies:
      position = enemy["position"]
      state[*position, 1] = True
      state[*position, 2] = True
      if enemy["kind"] == "jet":
        state[position[0], position[1], 5] = True
    for missile in self.missiles:
      state[*missile["position"], 1] = True
    for shot in self.shots:
      state[*shot["position"], 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_threats:
      state[*position, 4] = True
    for truck in self.trucks:
      state[*truck, 2] = True
    state[9, : min(10, len(self.trucks)), 5] = True
    state[8, : self.lives, 5] = True
    state[7, : self.wave, 5] = True
    return state
