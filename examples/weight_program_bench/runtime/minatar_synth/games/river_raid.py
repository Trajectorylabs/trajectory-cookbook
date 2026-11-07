import numpy as np


OBJECT_VALUES = {"tanker": 30, "helicopter": 60, "fuel": 80, "jet": 100, "bridge": 500}
SPAWN_OFFSETS = {"tanker": 4, "helicopter": 9, "fuel": 15, "jet": 21, "bridge": 29}


class RiverRaidEnvironment:
  """Sixteen bridge sections with fuel, throttle, traffic, and checkpoints."""

  score_cap = 12_320.0

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
    self.section = 0
    self.section_progress = 0
    self.player_column = 5
    self.previous_player_column = 5
    self.speed = 1
    self.fuel = 100
    self.lives = 4
    self.score = 0.0
    self.next_reinforcement = 10_000
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed: set[tuple[int, str]] = set()
    self.missiles: list[list[int]] = []
    self.previous_missiles: list[list[int]] = []
    self._load_section()

  def _load_section(self) -> None:
    self.objects = []
    for kind, offset in SPAWN_OFFSETS.items():
      if (self.section, kind) in self.claimed:
        continue
      column = 2 + (self.section * 3 + len(kind)) % 6
      self.objects.append({"kind": kind, "offset": offset, "column": column, "direction": 1 if offset % 2 else -1})
    self.previous_objects = {str(item["kind"]): self._object_position(item) for item in self.objects}

  def _object_position(self, item: dict[str, object]) -> list[int]:
    return [8 - (int(item["offset"]) - self.section_progress), int(item["column"])]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player_column = self.player_column
    self.previous_missiles = [missile.copy() for missile in self.missiles]
    self.previous_objects = {str(item["kind"]): self._object_position(item) for item in self.objects}
    if action == 0:
      self.speed = 1
    elif action == 1:
      self.player_column = max(0, self.player_column - 1)
      for missile in self.missiles:
        missile[1] = max(0, missile[1] - 1)
    elif action == 3:
      self.player_column = min(9, self.player_column + 1)
      for missile in self.missiles:
        missile[1] = min(9, missile[1] + 1)
    elif action == 2:
      self.speed = 2
    elif action == 4:
      self.speed = 0
    elif action == 5 and len(self.missiles) < 3:
      self.missiles.append([8, self.player_column])
    section_before = self.section
    reward = self._move_missiles()
    if self.section == section_before and not self.terminal:
      self._move_objects()
      self._refuel()
      self.section_progress += self.speed
      self.fuel -= 1
      self._resolve_collisions()
    self.score += reward
    while self.score >= self.next_reinforcement:
      self.lives = min(10, self.lives + 1)
      self.next_reinforcement += 10_000
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_missiles(self) -> float:
    reward = 0.0
    for missile in list(self.missiles):
      missile[0] -= 1
      hit = None
      for item in self.objects:
        if missile == self._object_position(item):
          hit = item
          break
      if hit is not None:
        kind = str(hit["kind"])
        self.claimed.add((self.section, kind))
        self.objects.remove(hit)
        self.missiles.remove(missile)
        reward += float(OBJECT_VALUES[kind])
        if kind == "bridge":
          self.section += 1
          if self.section >= 16:
            self.terminal = True
          else:
            self.section_progress = 0
            self.missiles.clear()
            self._load_section()
          break
      elif missile[0] < 0:
        self.missiles.remove(missile)
    return reward

  def _move_objects(self) -> None:
    for item in self.objects:
      if item["kind"] not in ("helicopter", "jet"):
        continue
      interval = max(1, 4 - self.section // 5)
      if self.steps % interval:
        continue
      item["column"] = int(item["column"]) + int(item["direction"])
      if item["column"] in (1, 8):
        item["direction"] = -int(item["direction"])

  def _refuel(self) -> None:
    for item in self.objects:
      if item["kind"] == "fuel" and self._object_position(item) == [8, self.player_column]:
        self.fuel = min(100, self.fuel + (8 if self.speed == 0 else 4))

  def _river_bounds(self, progress: int) -> tuple[int, int]:
    center = 5 + ((progress // 7 + self.section) % 3 - 1)
    half_width = max(2, 4 - self.section // 6)
    return max(0, center - half_width), min(9, center + half_width)

  def _resolve_collisions(self) -> None:
    left, right = self._river_bounds(self.section_progress)
    object_hit = any(self._object_position(item) == [8, self.player_column] for item in self.objects)
    if self.player_column <= left or self.player_column >= right or object_hit or self.fuel <= 0:
      self._lose_jet()

  def _lose_jet(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    self.section_progress = 0
    self.player_column = 5
    self.previous_player_column = 5
    self.speed = 1
    self.fuel = 100
    self.missiles.clear()
    self._load_section()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, self.player_column, 0] = True
    state[8, self.previous_player_column, 4] = True
    for row in range(9):
      left, right = self._river_bounds(self.section_progress + 8 - row)
      state[row, : left + 1, 3] = True
      state[row, right:, 3] = True
    kind_codes = {"tanker": 0, "helicopter": 1, "fuel": 2, "jet": 3}
    for item in self.objects:
      row, column = self._object_position(item)
      if 0 <= row < 9:
        if item["kind"] == "bridge":
          left, right = self._river_bounds(self.section_progress + 8 - row)
          state[row, left + 1 : right, 1] = True
        else:
          code = kind_codes[str(item["kind"])]
          state[row, column, 1] = True
          state[row, column, 2] = bool(code & 1)
          state[row, column, 3] = bool(code & 2)
        previous = self.previous_objects.get(str(item["kind"]))
        if previous is not None and 0 <= previous[0] < 9:
          state[*previous, 4] = True
    for missile in self.missiles:
      state[*missile, 2] = True
    for missile in self.previous_missiles:
      state[*missile, 4] = True
    for kind_index, kind in enumerate(OBJECT_VALUES):
      state[0, kind_index, 5] = (self.section, kind) in self.claimed
    for bit in range(4):
      state[0, 5 + bit, 5] = bool(self.section & (1 << bit))
      state[1, bit, 5] = bool(self.lives & (1 << bit))
    for bit in range(7):
      index = 14 + bit
      state[index // 10, index % 10, 5] = bool(max(0, self.fuel) & (1 << bit))
    state[2, 1, 5] = bool(self.speed & 1)
    state[2, 2, 5] = bool(self.speed & 2)
    state[2, 3, 5] = self.score >= 10_000
    for bit in range(5):
      state[2, 4 + bit, 5] = bool(self.section_progress & (1 << bit))
    for bit in range(12):
      index = 30 + bit
      state[index // 10, index % 10, 5] = bool(self.steps & (1 << bit))
    return state
