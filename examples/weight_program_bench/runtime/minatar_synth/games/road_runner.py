import numpy as np


SEED_OFFSETS = tuple(3 + index * 3 for index in range(10))
HAZARDS = {8: "mine", 16: "truck", 24: "cannon", 34: "rock", 38: "cliff"}


class RoadRunnerEnvironment:
  """Eight desert chases with seed streaks and Coyote hazard baiting."""

  score_cap = 57_600.0

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
    self.level = 0
    self.lives = 3
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed: set[tuple[int, str, int]] = set()
    self._start_level()

  def _start_level(self) -> None:
    self.progress = 0
    self.lane = 5
    self.previous_lane = self.lane
    self.jump_timer = 0
    self.magnet_timer = 0
    self.seed_streak = 0
    self.coyote_distance = 8
    self.previous_coyote_distance = self.coyote_distance
    self.projectile: list[int] | None = None
    self.previous_projectile: list[int] | None = None
    self.level_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_lane = self.lane
    self.previous_coyote_distance = self.coyote_distance
    self.previous_projectile = None if self.projectile is None else self.projectile.copy()
    progress_before = self.progress
    if action == 1:
      self.progress += 1
      self.coyote_distance = min(9, self.coyote_distance + 2)
    elif action == 3:
      self.progress = max(0, self.progress - 1)
      self.coyote_distance -= 1
    elif action == 2:
      self.lane = max(3, self.lane - 1)
    elif action == 4:
      self.lane = min(7, self.lane + 1)
    elif action == 5:
      self.jump_timer = 3
    reward = self._resolve_course(progress_before)
    self._move_projectile()
    self.coyote_distance -= 1 + int(self.magnet_timer > 0)
    reward += self._resolve_coyote_hazard()
    self._resolve_player_hazards()
    self.score += reward
    self.steps += 1
    self.level_steps += 1
    self.jump_timer = max(0, self.jump_timer - 1)
    self.magnet_timer = max(0, self.magnet_timer - 1)
    if self.progress >= 40 and not self.terminal:
      self.level += 1
      if self.level >= 8:
        self.terminal = True
      else:
        self._start_level()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _resolve_course(self, previous_progress: int) -> float:
    reward = 0.0
    for index, offset in enumerate(SEED_OFFSETS):
      identity = (self.level, "seed", index)
      if identity in self.claimed or not previous_progress < offset <= self.progress:
        continue
      self.claimed.add(identity)
      seed_lane = 4 + (index + self.level) % 4
      if self.lane == seed_lane:
        self.seed_streak = min(10, self.seed_streak + 1)
        reward += float(self.seed_streak * 100)
      else:
        self.seed_streak = 0
    steel_identity = (self.level, "steel", 0)
    if steel_identity not in self.claimed and previous_progress < 32 <= self.progress:
      self.claimed.add(steel_identity)
      if self.lane == 5:
        self.magnet_timer = 20
        reward += 100.0
    return reward

  def _move_projectile(self) -> None:
    if self.progress >= 23 and self.progress <= 28 and self.projectile is None:
      self.projectile = [self.lane, 9]
    if self.projectile is not None:
      self.projectile[1] -= 1 + self.level // 3
      if self.projectile[1] < 0:
        self.projectile = None

  def _resolve_coyote_hazard(self) -> float:
    if self.coyote_distance > 2:
      return 0.0
    for offset, kind in HAZARDS.items():
      if kind == "cliff" or abs(self.progress - offset) > 1:
        continue
      identity = (self.level, "coyote", offset)
      if identity in self.claimed:
        continue
      hazard_lane = 4 + (offset // 8 + self.level) % 4
      if self.lane == hazard_lane and self.jump_timer == 0:
        continue
      self.claimed.add(identity)
      self.coyote_distance = 8
      return 1_000.0 if kind == "truck" else 200.0
    return 0.0

  def _resolve_player_hazards(self) -> None:
    if self.coyote_distance <= 0:
      self._lose_life()
      return
    for offset, kind in HAZARDS.items():
      if abs(self.progress - offset) > 0:
        continue
      hazard_lane = 4 + (offset // 8 + self.level) % 4
      if self.lane != hazard_lane or self.jump_timer > 0:
        continue
      self._lose_life()
      return
    if self.projectile is not None and self.projectile == [self.lane, 5] and self.jump_timer == 0:
      self._lose_life()

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    self.lane = 5
    self.jump_timer = 0
    self.magnet_timer = 0
    self.coyote_distance = 8
    self.projectile = None

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.lane, 5, 0] = True
    state[self.previous_lane, 5, 4] = True
    coyote_column = max(0, 5 - self.coyote_distance)
    previous_coyote_column = max(0, 5 - self.previous_coyote_distance)
    state[self.lane, coyote_column, 1] = True
    state[self.previous_lane, previous_coyote_column, 4] = True
    for index, offset in enumerate(SEED_OFFSETS):
      screen_column = 5 + offset - self.progress
      if 0 <= screen_column < 10 and (self.level, "seed", index) not in self.claimed:
        lane = 4 + (index + self.level) % 4
        state[lane, screen_column, 2] = True
    steel_column = 5 + 32 - self.progress
    if 0 <= steel_column < 10 and (self.level, "steel", 0) not in self.claimed:
      state[5, steel_column, 2] = True
      state[5, steel_column, 3] = True
    for offset, kind in HAZARDS.items():
      screen_column = 5 + offset - self.progress
      if 0 <= screen_column < 10:
        lane = 4 + (offset // 8 + self.level) % 4
        state[lane, screen_column, 1] = True
        if kind == "cliff":
          state[3:8, screen_column, 3] = True
    if self.projectile is not None:
      state[*self.projectile, 1] = True
    if self.previous_projectile is not None:
      state[*self.previous_projectile, 4] = True
    for index in range(10):
      state[0, index, 5] = (self.level, "seed", index) in self.claimed
    state[1, 0, 5] = (self.level, "steel", 0) in self.claimed
    for index, offset in enumerate(HAZARDS):
      if HAZARDS[offset] != "cliff":
        state[1, 1 + index, 5] = (self.level, "coyote", offset) in self.claimed
    for bit in range(3):
      state[2, bit, 5] = bool(self.level & (1 << bit))
      state[2, 3 + bit, 5] = bool(self.lives & (1 << bit))
    for bit in range(6):
      state[3, bit, 5] = bool(self.progress & (1 << bit))
    for bit in range(4):
      state[3, 6 + bit, 5] = bool(self.seed_streak & (1 << bit))
    for bit in range(5):
      state[4, bit, 5] = bool(self.magnet_timer & (1 << bit))
    for bit in range(2):
      state[4, 5 + bit, 5] = bool(self.jump_timer & (1 << bit))
    for bit in range(4):
      index = 47 + bit
      state[index // 10, index % 10, 5] = bool(self.coyote_distance & (1 << bit))
    return state
