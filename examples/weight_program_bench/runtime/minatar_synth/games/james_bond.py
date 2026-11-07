import numpy as np


MISSION_NAMES = ("diamonds", "moonraker", "spy")


class JamesBondEnvironment:
  """Three-mission multipurpose-craft campaign with alternating weapons."""

  score_cap = 20_500.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_800):
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
    self.mission = 0
    self.lives = 6
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.destroyed: set[tuple[int, str, int]] = set()
    self._start_mission()

  def _start_mission(self) -> None:
    self.player = [5, 1]
    self.previous_player = self.player.copy()
    self.entities: list[dict[str, object]] = []
    self.projectiles: list[dict[str, object]] = []
    self.mission_steps = 0
    self.next_weapon = "shot"
    self.light_timer = 0
    self.rig: list[int] | None = None
    self.lab: list[int] | None = None
    self.capsule: list[int] | None = None
    self.capsule_timer = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    mission_before = self.mission
    lives_before = self.lives
    self._move_player(action)
    if action == 5:
      self._fire_weapon()
    self._spawn_entities()
    reward = self._move_projectiles()
    if self.terminal or self.mission >= 3:
      return self._finish_step(reward)
    self._move_entities()
    self._resolve_player_contacts()
    if self.terminal or self.lives != lives_before:
      return self._finish_step(reward, advance_mission=False)
    reward += self._resolve_objectives()
    return self._finish_step(reward, advance_mission=self.mission == mission_before)

  def _finish_step(self, reward: float, advance_mission: bool = True) -> tuple[float, bool]:
    self.score += reward
    self.steps += 1
    if self.mission < 3 and advance_mission:
      self.mission_steps += 1
    self.light_timer = max(0, self.light_timer - 1)
    if self.capsule is not None:
      self.capsule_timer -= 1
      if self.capsule_timer <= 0:
        self.capsule = None
        self.lab = None
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if action == 1:
      self.player[1] = max(0, self.player[1] - 1)
    elif action == 2:
      self.player[0] = max(0, self.player[0] - 1)
    elif action == 3:
      self.player[1] = min(9, self.player[1] + 1)
    elif action == 4:
      self.player[0] = min(9, self.player[0] + 1)
    elif self.player[0] < 5:
      self.player[0] += 1
    elif self.player[0] > 5:
      self.player[0] -= 1

  def _fire_weapon(self) -> None:
    if len(self.projectiles) >= 2:
      return
    if self.next_weapon == "shot":
      self.projectiles.append({"position": self.player.copy(), "direction": [-1, 1], "kind": "shot"})
      self.next_weapon = "bomb"
    else:
      self.projectiles.append({"position": self.player.copy(), "direction": [1, 1], "kind": "bomb"})
      self.next_weapon = "shot"

  def _mission_schedule(self) -> list[tuple[int, str, int, int, int]]:
    if self.mission == 0:
      return [
        (5, "diamond", 1, 50, 4),
        (18, "frogman", 7, 200, 4),
        (32, "diamond", 1, 50, 4),
        (46, "frogman", 7, 200, 4),
      ]
    if self.mission == 1:
      return [
        (5, "satellite", 1, 500, 3),
        (15, "shuttle", 8, 200, 4),
        (28, "submarine", 7, 200, 3),
        (40, "torpedo", 6, 100, 3),
      ]
    return [
      (8, "rocket", 8, 100, 4),
      (24, "submarine", 7, 200, 3),
      (38, "torpedo", 6, 100, 3),
    ]

  def _spawn_entities(self) -> None:
    cycle_step = self.mission_steps % 60
    cycle = self.mission_steps // 60
    for spawn_step, kind, row, value, count in self._mission_schedule():
      index = cycle % count
      identity = (self.mission, kind, index)
      present = any(entity["identity"] == identity for entity in self.entities)
      if cycle_step == spawn_step and identity not in self.destroyed and not present:
        self.entities.append(
          {
            "identity": identity,
            "kind": kind,
            "position": [row, 9],
            "value": value,
            "age": 0,
          }
        )
    if self.mission == 0 and self.mission_steps >= 120 and self.rig is None:
      self.rig = [3, 8]
    if self.mission == 2 and self.mission_steps >= 120 and self.lab is None and self.capsule is None:
      self.lab = [8, 8]

  def _move_projectiles(self) -> float:
    reward = 0.0
    remaining = []
    for projectile in self.projectiles:
      position = projectile["position"]
      direction = projectile["direction"]
      position[0] += direction[0]
      position[1] += direction[1]
      hit = False
      for entity in list(self.entities):
        if entity["position"] == position:
          self.entities.remove(entity)
          identity = entity["identity"]
          if identity not in self.destroyed:
            self.destroyed.add(identity)
            reward += float(entity["value"])
            if entity["kind"] == "diamond":
              self.light_timer = 30
          hit = True
          break
      if self.lab == position and projectile["kind"] == "bomb":
        self.lab = None
        self.capsule = [6, 8]
        self.capsule_timer = 8
        hit = True
      elif self.capsule == position and projectile["kind"] == "bomb":
        self.capsule[0] = min(8, self.capsule[0] + 1)
        self.capsule_timer = 8
        hit = True
      if not hit and 0 <= position[0] < 10 and 0 <= position[1] < 10:
        remaining.append(projectile)
    self.projectiles = remaining
    return reward

  def _move_entities(self) -> None:
    for entity in list(self.entities):
      entity["age"] = int(entity["age"]) + 1
      position = entity["position"]
      position[1] -= 1
      if entity["kind"] in ("shuttle", "rocket") and position[0] <= 1:
        entity["kind"] = "poison" if entity["kind"] == "shuttle" else "fire_bomb"
        entity["value"] = 100
        position[0] = 1
      elif entity["kind"] in ("shuttle", "rocket"):
        position[0] -= 1
      elif entity["kind"] in ("poison", "fire_bomb"):
        position[0] += 1
      if position[1] < 0 or position[0] > 9:
        self.entities.remove(entity)

  def _resolve_player_contacts(self) -> None:
    if any(entity["position"] == self.player for entity in self.entities):
      self._lose_life()

  def _resolve_objectives(self) -> float:
    if self.mission == 0 and self.rig is not None and self.player == self.rig and self.light_timer > 0:
      return self._complete_mission()
    satellites = sum(1 for mission, kind, _ in self.destroyed if mission == 1 and kind == "satellite")
    if self.mission == 1 and satellites >= 3:
      return self._complete_mission()
    if self.mission == 2 and self.capsule is not None and self.player == self.capsule:
      return self._complete_mission()
    return 0.0

  def _complete_mission(self) -> float:
    self.mission += 1
    if self.mission >= 3:
      self.terminal = True
    else:
      self._start_mission()
    return 5_000.0

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self._start_mission()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for entity in self.entities:
      channel = 2 if entity["kind"] in ("poison", "fire_bomb") else 1
      state[*entity["position"], channel] = True
    if self.rig is not None and self.light_timer > 0:
      state[*self.rig, 2] = True
    if self.lab is not None:
      state[*self.lab, 2] = True
    if self.capsule is not None:
      state[*self.capsule, 2] = True
    for projectile in self.projectiles:
      state[*projectile["position"], 3] = True
    state[*self.previous_player, 4] = True
    reward_identities = []
    for mission in range(3):
      old_mission = self.mission
      self.mission = mission
      for _, kind, _, _, count in self._mission_schedule():
        reward_identities.extend((mission, kind, index) for index in range(count))
      self.mission = old_mission
    for index, identity in enumerate(reward_identities):
      state[index // 10, index % 10, 5] = identity in self.destroyed
    for bit in range(2):
      state[4, 4 + bit, 5] = bool(self.mission & (1 << bit))
    state[4, 6, 5] = self.next_weapon == "shot"
    state[4, 7, 5] = self.next_weapon == "bomb"
    for bit in range(3):
      state[5, bit, 5] = bool(self.lives & (1 << bit))
    for bit in range(4):
      state[5, 3 + bit, 5] = bool(self.capsule_timer & (1 << bit))
    for bit in range(5):
      state[6, bit, 5] = bool(self.light_timer & (1 << bit))
    cycle_step = self.mission_steps % 60
    cycle = (self.mission_steps // 60) % 12
    for bit in range(6):
      state[7, bit, 5] = bool(cycle_step & (1 << bit))
    state[7, 6, 5] = self.mission_steps >= 120
    for bit in range(4):
      state[9, 3 + bit, 5] = bool(cycle & (1 << bit))
    for bit in range(10):
      state[8, bit, 5] = bool(self.steps & (1 << bit))
    state[9, 9, 5] = bool(self.steps & (1 << 10))
    for index, name in enumerate(MISSION_NAMES):
      state[4, index, 5] = self.mission < 3 and MISSION_NAMES[self.mission] == name
    return state
