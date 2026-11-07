import numpy as np


BOSS_NAMES = ("fist_fighter", "boomerang_thrower", "giant_kicker", "lightning_magician", "gang_master")
BOSS_VALUES = (2_000, 3_000, 3_000, 5_000, 10_000)
REGULAR_ROSTERS = (
  ("henchman",) * 4 + ("knife_thrower",) * 2,
  ("henchman",) * 3 + ("midget",) * 3 + ("knife_thrower",),
  ("dragon_ball",) * 3 + ("snake",) * 2 + ("moth",) * 2,
  ("moth",) * 3 + ("knife_thrower",) * 2 + ("henchman",) * 2,
  ("henchman",) * 4 + ("knife_thrower",) * 2 + ("midget",) * 2,
)
ENEMY_VALUES = {
  "henchman": (100, 100),
  "knife_thrower": (500, 800),
  "midget": (200, 300),
  "dragon_ball": (2_000, 2_000),
  "moth": (500, 600),
  "snake": (0, 0),
}
ENEMY_HEALTH = {"henchman": 1, "knife_thrower": 2, "midget": 1, "dragon_ball": 1, "moth": 1, "snake": 99}


class KungFuMasterEnvironment:
  """Two five-floor temple climbs with stance-sensitive combat."""

  score_cap = 120_800.0

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
    self.climb = 0
    self.floor = 0
    self.lives = 3
    self.score = 0.0
    self.bonus_life_awarded = False
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed: set[tuple[int, int, int]] = set()
    self._start_floor()

  def _start_floor(self) -> None:
    self.player = [5, 5]
    self.previous_player = self.player.copy()
    self.facing = 1
    self.energy = 10
    self.timer = 300
    self.jump_timer = 0
    self.squat_timer = 0
    self.grabbed_by: int | None = None
    self.last_escape_direction = 0
    self.enemies: dict[int, dict[str, object]] = {}
    for index, kind in enumerate(REGULAR_ROSTERS[self.floor]):
      if (self.climb, self.floor, index) not in self.claimed:
        side = 9 if index % 2 == 0 else 0
        row = 4 if kind == "moth" else 5
        self.enemies[index] = {
          "kind": kind,
          "position": [row, side],
          "health": ENEMY_HEALTH[kind],
          "cooldown": index % 4,
          "direction": -1 if side == 9 else 1,
        }
    self.knives: list[dict[str, object]] = []
    self.previous_knives: list[list[int]] = []
    self.boss: dict[str, object] | None = None
    self.boss_projectiles: list[list[int]] = []
    self.floor_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    floor_before = (self.climb, self.floor)
    lives_before = self.lives
    self._apply_stance_and_movement(action)
    reward = self._attack() if action == 5 else 0.0
    self._move_regular_enemies()
    self._move_knives()
    self._move_boss()
    self._resolve_damage()
    if self.lives != lives_before or self.terminal:
      return self._finish_step(reward, consume_timer=False)
    if not self.enemies and self.boss is None:
      self._spawn_boss()
    transitioned = (self.climb, self.floor) != floor_before
    return self._finish_step(reward, consume_timer=not transitioned)

  def _finish_step(self, reward: float, consume_timer: bool = True) -> tuple[float, bool]:
    self.score += reward
    if self.score >= 75_000 and not self.bonus_life_awarded:
      self.lives = min(9, self.lives + 1)
      self.bonus_life_awarded = True
    if consume_timer:
      self.timer -= 1
      self.floor_steps += 1
      self.jump_timer = max(0, self.jump_timer - 1)
      self.squat_timer = max(0, self.squat_timer - 1)
      if self.timer <= 0:
        self._lose_life()
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _apply_stance_and_movement(self, action: int) -> None:
    if self.grabbed_by is not None:
      direction = -1 if action == 1 else 1 if action == 3 else 0
      if direction and direction != self.last_escape_direction:
        self.last_escape_direction = direction
        enemy = self.enemies.get(self.grabbed_by)
        if enemy is not None:
          enemy["health"] = int(enemy["health"]) - 1
          if enemy["health"] <= -2:
            self.grabbed_by = None
      return
    if action == 1:
      self.player[1] = max(0, self.player[1] - 1)
      self.facing = -1
    elif action == 3:
      self.player[1] = min(9, self.player[1] + 1)
      self.facing = 1
    elif action == 2:
      self.jump_timer = 2
      self.player[0] = 3
    elif action == 4:
      self.squat_timer = 2
      self.player[0] = 6
    elif self.jump_timer == 0 and self.squat_timer == 0:
      self.player[0] = 5

  def _attack(self) -> float:
    if self.jump_timer > 0:
      return 0.0
    low = self.squat_timer > 0
    candidates = []
    for index, enemy in self.enemies.items():
      position = enemy["position"]
      distance = abs(position[1] - self.player[1])
      correct_side = int(np.sign(position[1] - self.player[1])) == self.facing
      height_ok = low if enemy["kind"] == "midget" else not low
      if correct_side and distance <= 2 and height_ok:
        candidates.append((distance, index, enemy))
    if candidates:
      distance, index, enemy = min(candidates)
      if enemy["kind"] == "snake":
        return 0.0
      enemy["health"] = int(enemy["health"]) - 1
      if enemy["health"] <= 0:
        del self.enemies[index]
        self.claimed.add((self.climb, self.floor, index))
        kick_value, punch_value = ENEMY_VALUES[str(enemy["kind"])]
        return float(kick_value if distance == 2 else punch_value)
      return 0.0
    if self.boss is not None:
      distance = abs(self.boss["position"][1] - self.player[1])
      correct_side = int(np.sign(self.boss["position"][1] - self.player[1])) == self.facing
      if correct_side and distance <= 2:
        self.boss["health"] = int(self.boss["health"]) - 1
        if self.boss["health"] <= 0:
          return self._complete_floor()
    return 0.0

  def _move_regular_enemies(self) -> None:
    interval = max(1, 3 - self.climb)
    for index, enemy in list(self.enemies.items()):
      kind = str(enemy["kind"])
      position = enemy["position"]
      if kind == "snake":
        position[1] += int(enemy["direction"])
        if not 0 <= position[1] < 10:
          del self.enemies[index]
        continue
      enemy["cooldown"] = int(enemy["cooldown"]) - 1
      if kind == "knife_thrower" and enemy["cooldown"] <= 0:
        height = "high" if (index + self.floor_steps) % 2 else "low"
        self.knives.append(
          {"position": position.copy(), "direction": int(np.sign(self.player[1] - position[1])) or -1, "height": height}
        )
        enemy["cooldown"] = max(3, 7 - self.climb)
      if self.floor_steps % interval:
        continue
      if kind == "moth":
        position[0] = 4 if position[0] == 5 else 5
      position[1] += int(np.sign(self.player[1] - position[1]))
      if kind == "snake" and position[1] == self.player[1]:
        direction = -1 if position[1] > 0 else 1
        position[1] += direction

  def _move_knives(self) -> None:
    self.previous_knives = [list(knife["position"]) for knife in self.knives]
    for knife in list(self.knives):
      knife["position"][1] += int(knife["direction"])
      if not 0 <= knife["position"][1] < 10:
        self.knives.remove(knife)

  def _spawn_boss(self) -> None:
    self.boss = {
      "kind": BOSS_NAMES[self.floor],
      "position": [5, 8],
      "health": 4 + self.floor + self.climb,
      "cooldown": 3,
    }

  def _move_boss(self) -> None:
    if self.boss is None:
      return
    self.boss["cooldown"] = int(self.boss["cooldown"]) - 1
    if self.boss["cooldown"] <= 0:
      self.boss_projectiles.append(self.boss["position"].copy())
      self.boss["cooldown"] = max(2, 6 - self.floor // 2 - self.climb)
    if self.floor_steps % 2 == 0:
      self.boss["position"][1] += int(np.sign(self.player[1] - self.boss["position"][1]))
    for projectile in list(self.boss_projectiles):
      projectile[1] += int(np.sign(self.player[1] - projectile[1]))
      if not 0 <= projectile[1] < 10:
        self.boss_projectiles.remove(projectile)

  def _resolve_damage(self) -> None:
    for index, enemy in self.enemies.items():
      if enemy["position"] != self.player:
        continue
      if enemy["kind"] == "henchman" and self.grabbed_by is None:
        self.grabbed_by = index
      self.energy -= 1
      break
    for knife in list(self.knives):
      if knife["position"][1] != self.player[1]:
        continue
      avoided = (knife["height"] == "low" and self.jump_timer > 0) or (
        knife["height"] == "high" and self.squat_timer > 0
      )
      if not avoided:
        self.energy -= 2
      self.knives.remove(knife)
    if self.boss is not None and self.boss["position"] == self.player:
      self.energy -= 3
    for projectile in list(self.boss_projectiles):
      if projectile[1] == self.player[1]:
        self.energy -= 2
        self.boss_projectiles.remove(projectile)
    if self.grabbed_by is not None:
      self.energy -= 1
    if self.energy <= 0:
      self._lose_life()

  def _complete_floor(self) -> float:
    reward = float(BOSS_VALUES[self.floor] + self.timer * 10 + self.energy * 100)
    self.floor += 1
    if self.floor >= 5:
      self.floor = 0
      self.climb += 1
    if self.climb >= 2:
      self.terminal = True
    else:
      self._start_floor()
    return reward

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self._start_floor()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies.values():
      state[*enemy["position"], 1] = True
      kind = str(enemy["kind"])
      if kind in ("knife_thrower", "dragon_ball"):
        state[*enemy["position"], 2] = True
      if kind in ("midget", "dragon_ball"):
        state[*enemy["position"], 3] = True
      if kind in ("moth", "snake"):
        state[*enemy["position"], 4] = True
    if self.boss is not None:
      state[*self.boss["position"], 1] = True
    for knife in self.knives:
      state[*knife["position"], 2 if knife["height"] == "high" else 1] = True
    for projectile in self.boss_projectiles:
      state[*projectile, 1] = True
    state[*self.previous_player, 4] = True
    for knife in self.previous_knives:
      state[*knife, 4] = True
    for index in range(len(REGULAR_ROSTERS[self.floor])):
      state[index // 10, index % 10, 5] = (self.climb, self.floor, index) in self.claimed
    state[2, self.floor, 5] = True
    state[2, 5 + self.climb, 5] = True
    for bit in range(4):
      state[3, bit, 5] = bool(self.lives & (1 << bit))
      state[3, 4 + bit, 5] = bool(self.energy & (1 << bit))
    for bit in range(9):
      state[4, bit, 5] = bool(self.timer & (1 << bit))
    for bit in range(12):
      state[5 + bit // 10, bit % 10, 5] = bool(self.steps & (1 << bit))
    state[7, 0, 5] = self.facing > 0
    state[7, 1 : 1 + self.jump_timer, 5] = True
    state[7, 3 : 3 + self.squat_timer, 5] = True
    state[7, 5, 5] = self.grabbed_by is not None
    if self.grabbed_by is not None:
      for bit in range(4):
        state[7, 6 + bit, 5] = bool(self.grabbed_by & (1 << bit))
    if self.boss is not None:
      for bit in range(4):
        state[8, bit, 5] = bool(int(self.boss["health"]) & (1 << bit))
      for bit in range(3):
        state[8, 4 + bit, 5] = bool(int(self.boss["cooldown"]) & (1 << bit))
    state[8, 7, 5] = self.bonus_life_awarded
    knife_throwers = [enemy for enemy in self.enemies.values() if enemy["kind"] == "knife_thrower"]
    for index, enemy in enumerate(knife_throwers[:2]):
      for bit in range(3):
        state[1, index * 3 + bit, 5] = bool(int(enemy["cooldown"]) & (1 << bit))
    for bit in range(10):
      state[9, bit, 5] = bool(self.floor_steps & (1 << bit))
    for bit in range(2):
      state[8, 8 + bit, 5] = bool(self.floor_steps & (1 << (10 + bit)))
    return state
