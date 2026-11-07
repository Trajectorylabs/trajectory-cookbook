import numpy as np


ALIEN_VALUES = {
  "pod": 1_000.0,
  "swarmer": 500.0,
  "bomber": 250.0,
  "baiter": 200.0,
  "lander": 150.0,
  "mutant": 150.0,
}


class DefenderEnvironment:
  """Five-wave scanner shooter centered on humanoid abduction and rescue."""

  score_cap = 48_500.0

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
    self.lives = 3
    self.smart_bombs = 3
    self.score = 0.0
    self.next_bonus_score = 10_000.0
    self.wave = 1
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.shots: list[dict[str, object]] = []
    self.hostile_shots: list[list[int]] = []
    self.humanoids = [
      {"position": [8, column], "state": "ground"} for column in (0, 2, 4, 6, 8)
    ]
    self.carried_humanoid: dict[str, object] | None = None
    self.previous_threats: list[list[int]] = []
    self._start_wave()

  def _start_wave(self) -> None:
    kinds = ["lander"] * 5 + ["bomber", "baiter", "pod"]
    self.enemies = [
      {"kind": kind, "position": [1 + index % 5, (index * 3 + self.wave) % 10]}
      for index, kind in enumerate(kinds)
    ]
    self.hostile_shots = []
    self.wave_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_threats = self._threat_positions()
    reward = self._apply_action(action)
    self._move_world()
    reward += self._resolve_player_shots()
    reward += self._resolve_humanoids()
    reward += self._resolve_player_contact()
    self.score += reward
    if not self.enemies:
      reward += self._complete_wave()
    while self.score >= self.next_bonus_score:
      self.lives += 1
      self.smart_bombs += 1
      self.next_bonus_score += 10_000.0
    self.wave_steps += 1
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _apply_action(self, action: int) -> float:
    if action == 1:
      self.player[1] = (self.player[1] - 1) % 10
      self.facing = -1
    elif action == 2:
      self.player[0] = max(0, self.player[0] - 1)
    elif action == 3:
      self.player[1] = (self.player[1] + 1) % 10
      self.facing = 1
    elif action == 4:
      self.player[0] = min(8, self.player[0] + 1)
    elif action == 5:
      if self.player[0] == 0:
        self.player = [int(self.random.randint(1, 8)), int(self.random.randint(10))]
      elif self.player[0] == 8 and self.smart_bombs > 0:
        self.smart_bombs -= 1
        reward = sum(ALIEN_VALUES[str(enemy["kind"])] for enemy in self.enemies)
        destroyed_ids = {id(enemy) for enemy in self.enemies}
        for humanoid in self.humanoids:
          if humanoid["state"] == "abducted" and id(humanoid.get("lander")) in destroyed_ids:
            humanoid["state"] = "falling"
            humanoid.pop("lander", None)
        self.enemies = []
        return reward
      elif len(self.shots) < 3:
        self.shots.append({"position": self.player.copy(), "direction": self.facing})
    return 0.0

  def _move_world(self) -> None:
    for shot in self.shots:
      shot["position"][1] = (shot["position"][1] + int(shot["direction"])) % 10
      shot["distance"] = int(shot.get("distance", 0)) + 1
    self.shots = [shot for shot in self.shots if int(shot["distance"]) < 10]
    for enemy in self.enemies:
      kind = str(enemy["kind"])
      position = enemy["position"]
      if kind == "lander":
        self._move_lander(enemy)
      elif kind in ("baiter", "mutant", "swarmer"):
        position[0] += int(np.sign(self.player[0] - position[0]))
        position[1] = (position[1] + int(np.sign(self.player[1] - position[1]))) % 10
      else:
        position[1] = (position[1] - 1) % 10
      if self.random.rand() < min(0.2, 0.015 * self.wave):
        self.hostile_shots.append(list(position))
    for hostile_shot in self.hostile_shots:
      hostile_shot[0] += int(np.sign(self.player[0] - hostile_shot[0]))
      hostile_shot[1] = (hostile_shot[1] + int(np.sign(self.player[1] - hostile_shot[1]))) % 10
    self.hostile_shots = [shot for shot in self.hostile_shots if 0 <= shot[0] < 9]
    for humanoid in self.humanoids:
      if humanoid["state"] == "falling":
        humanoid["position"][0] += 1

  def _move_lander(self, lander: dict[str, object]) -> None:
    carried = next(
      (
        humanoid
        for humanoid in self.humanoids
        if humanoid["state"] == "abducted" and humanoid.get("lander") is lander
      ),
      None,
    )
    position = lander["position"]
    if carried is not None:
      position[0] -= 1
      carried["position"] = [position[0] + 1, position[1]]
      if position[0] <= 0:
        carried["state"] = "lost"
        lander["kind"] = "mutant"
        lander["position"] = [1, position[1]]
      return
    if self.carried_humanoid is not None or any(
      humanoid["state"] == "abducted" for humanoid in self.humanoids
    ):
      position[1] = (position[1] - 1) % 10
      return
    ground = [humanoid for humanoid in self.humanoids if humanoid["state"] == "ground"]
    if not ground:
      position[1] = (position[1] - 1) % 10
      return
    target = min(ground, key=lambda humanoid: abs(humanoid["position"][1] - position[1]))
    if position[1] != target["position"][1]:
      position[1] = (position[1] + int(np.sign(target["position"][1] - position[1]))) % 10
    elif position[0] < 7:
      position[0] += 1
    else:
      target["state"] = "abducted"
      target["lander"] = lander

  def _resolve_player_shots(self) -> float:
    reward = 0.0
    remaining_shots = []
    destroyed_ids: set[int] = set()
    spawned: list[dict[str, object]] = []
    for shot in self.shots:
      enemy = next(
        (
          candidate
          for candidate in self.enemies
          if id(candidate) not in destroyed_ids and candidate["position"] == shot["position"]
        ),
        None,
      )
      if enemy is None:
        remaining_shots.append(shot)
        continue
      destroyed_ids.add(id(enemy))
      reward += ALIEN_VALUES[str(enemy["kind"])]
      if enemy["kind"] == "pod":
        for offset in (-1, 0, 1, 2):
          spawned.append(
            {"kind": "swarmer", "position": [enemy["position"][0], (enemy["position"][1] + offset) % 10]}
          )
      for humanoid in self.humanoids:
        if humanoid["state"] == "abducted" and humanoid.get("lander") is enemy:
          humanoid["state"] = "falling"
          humanoid.pop("lander", None)
    self.shots = remaining_shots
    self.enemies = [enemy for enemy in self.enemies if id(enemy) not in destroyed_ids] + spawned
    return reward

  def _resolve_humanoids(self) -> float:
    reward = 0.0
    if self.carried_humanoid is None:
      caught = next(
        (
          humanoid
          for humanoid in self.humanoids
          if humanoid["state"] == "falling" and humanoid["position"] == self.player
        ),
        None,
      )
      if caught is not None:
        caught["state"] = "carried"
        self.carried_humanoid = caught
    if self.carried_humanoid is not None:
      self.carried_humanoid["position"] = self.player.copy()
      if self.player[0] == 8:
        self.carried_humanoid["state"] = "ground"
        self.carried_humanoid["position"] = [8, self.player[1]]
        self.carried_humanoid = None
        reward += 1_000.0
    for humanoid in self.humanoids:
      if humanoid["state"] != "falling" or humanoid["position"][0] < 8:
        continue
      if humanoid["position"][0] == 8:
        humanoid["state"] = "ground"
        reward += 250.0
      else:
        humanoid["state"] = "lost"
    return reward

  def _resolve_player_contact(self) -> float:
    collided_enemy = next(
      (enemy for enemy in self.enemies if enemy["position"] == self.player),
      None,
    )
    if collided_enemy is None and self.player not in self.hostile_shots:
      return 0.0
    reward = 0.0
    if collided_enemy is not None:
      reward = ALIEN_VALUES[str(collided_enemy["kind"])]
      self.enemies = [enemy for enemy in self.enemies if enemy is not collided_enemy]
      for humanoid in self.humanoids:
        if humanoid["state"] == "abducted" and humanoid.get("lander") is collided_enemy:
          humanoid["state"] = "falling"
          humanoid.pop("lander", None)
    self.lives -= 1
    self.player = [4, 1]
    self.previous_player = self.player.copy()
    self.shots = []
    self.hostile_shots = []
    if self.carried_humanoid is not None:
      self.carried_humanoid["state"] = "ground"
      self.carried_humanoid["position"] = [8, self.player[1]]
      self.carried_humanoid = None
    if self.lives <= 0:
      self.terminal = True
    return reward

  def _complete_wave(self) -> float:
    survivors = sum(humanoid["state"] in ("ground", "carried") for humanoid in self.humanoids)
    reward = survivors * 100.0
    if self.carried_humanoid is not None:
      self.carried_humanoid["state"] = "ground"
      self.carried_humanoid["position"] = [8, self.player[1]]
      self.carried_humanoid = None
    self.score += reward
    self.wave += 1
    if self.wave > 5:
      self.terminal = True
    else:
      if self.wave == 5 and survivors == 0:
        self.humanoids = [
          {"position": [8, column], "state": "ground"} for column in (0, 2, 4, 6, 8)
        ]
      self._start_wave()
    return reward

  def _threat_positions(self) -> list[list[int]]:
    return [list(enemy["position"]) for enemy in self.enemies] + [
      shot.copy() for shot in self.hostile_shots
    ]

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies:
      position = enemy["position"]
      state[*position, 1] = True
      state[*position, 2] = True
      if enemy["kind"] in ("lander", "mutant", "pod"):
        state[*position, 5] = True
    for hostile_shot in self.hostile_shots:
      state[*hostile_shot, 1] = True
    for shot in self.shots:
      state[*shot["position"], 3] = True
    for humanoid in self.humanoids:
      if humanoid["state"] != "lost":
        state[*humanoid["position"], 2] = True
        if humanoid["state"] != "ground":
          state[*humanoid["position"], 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_threats:
      state[*position, 4] = True
    state[9, : min(10, self.lives), 5] = True
    state[8, : min(10, self.smart_bombs), 5] = True
    state[7, : self.wave, 5] = True
    return state
