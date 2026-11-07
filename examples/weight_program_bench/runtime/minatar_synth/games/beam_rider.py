import numpy as np


BEAM_COLUMNS = (1, 3, 5, 7, 9)
SECTORS = (1, 2, 4, 6)


class BeamRiderEnvironment:
  """Five-beam sector defense with short lasers, scarce torpedoes, and rejuvenators."""

  score_cap = 34_200.0

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
    self.lane = 2
    self.lives = 3
    self.sector_index = 0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_sector()

  @property
  def sector(self) -> int:
    return SECTORS[self.sector_index]

  def _start_sector(self) -> None:
    self.torpedoes = 3
    self.saucers_destroyed = 0
    self.saucers_spawned = 0
    self.objects: list[dict[str, int | str | bool]] = []
    self.shots: list[dict[str, int | str]] = []
    self.previous_objects: list[list[int]] = []
    self.sector_step = 0
    self.sentinel: dict[str, int | str | bool] | None = None
    self.rejuvenator_spawned = False
    self.chirpers_spawned = 0
    self.hazard_spawned = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if action == 1:
      self.lane = max(0, self.lane - 1)
    elif action == 3:
      self.lane = min(4, self.lane + 1)
    elif action == 5:
      self.shots.append({"row": 8, "lane": self.lane, "kind": "laser", "ttl": 4})
    elif action == 2 and self.torpedoes > 0:
      self.torpedoes -= 1
      self.shots.append({"row": 8, "lane": self.lane, "kind": "torpedo", "ttl": 10})

    self._spawn_objects()
    self.previous_objects = [[int(obj["row"]), int(obj["lane"])] for obj in self.objects]
    self._move_objects()
    self._move_shots()
    reward = self._resolve_shots()
    self._resolve_player_contacts()
    reward += self._move_sentinel()
    self.sector_step += 1
    self.steps += 1
    if self.saucers_destroyed == 15 and self.sentinel is None:
      self.sentinel = {"row": 0, "column": 0, "kind": "sentinel", "active": True}
    if self.sentinel is not None and not bool(self.sentinel["active"]):
      self.sector_index += 1
      if self.sector_index == len(SECTORS):
        self.terminal = True
      else:
        self._start_sector()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _spawn_objects(self) -> None:
    if self.saucers_spawned < 15 and self.sector_step % 5 == 0:
      self.objects.append({"row": 0, "lane": int(self.random.randint(5)), "kind": "saucer", "red": False})
      self.saucers_spawned += 1
    if not self.rejuvenator_spawned and self.sector_step == 20:
      self.objects.append({"row": 0, "lane": int(self.random.randint(5)), "kind": "rejuvenator", "red": False})
      self.rejuvenator_spawned = True
    if self.sector >= 4 and self.chirpers_spawned < 3 and self.sector_step in (12, 30, 48):
      self.objects.append({"row": 0, "lane": int(self.random.randint(5)), "kind": "chirper", "red": False})
      self.chirpers_spawned += 1
    if self.sector >= 2 and not self.hazard_spawned and self.sector_step == 10:
      kind = "debris" if self.sector < 6 else "blocker"
      self.objects.append({"row": 0, "lane": int(self.random.randint(5)), "kind": kind, "red": False})
      self.hazard_spawned = True

  def _move_objects(self) -> None:
    for obj in self.objects:
      obj["row"] = int(obj["row"]) + 1
      if obj["kind"] == "chirper" and int(obj["row"]) % 2 == 0:
        obj["lane"] = int(np.clip(int(obj["lane"]) + self.random.choice((-1, 1)), 0, 4))
    for obj in self.objects:
      if obj["kind"] == "saucer" and int(obj["row"]) > 9:
        obj["row"] = 0
        obj["lane"] = int(self.random.randint(5))
    self.objects = [obj for obj in self.objects if int(obj["row"]) <= 9 or obj["kind"] == "saucer"]

  def _move_shots(self) -> None:
    for shot in self.shots:
      shot["row"] = int(shot["row"]) - 1
      shot["ttl"] = int(shot["ttl"]) - 1
    self.shots = [shot for shot in self.shots if int(shot["row"]) >= 0 and int(shot["ttl"]) > 0]

  def _resolve_shots(self) -> float:
    reward = 0.0
    remaining = []
    for shot in self.shots:
      hit = next(
        (obj for obj in self.objects if int(obj["row"]) == int(shot["row"]) and int(obj["lane"]) == int(shot["lane"])),
        None,
      )
      if hit is None:
        remaining.append(shot)
        continue
      kind = str(hit["kind"])
      torpedo = shot["kind"] == "torpedo"
      laser_vulnerable = kind in ("saucer", "chirper")
      if kind == "rejuvenator" and not bool(hit["red"]):
        hit["red"] = True
        continue
      if torpedo or laser_vulnerable:
        self.objects.remove(hit)
        if kind == "saucer":
          self.saucers_destroyed += 1
          reward += 100.0 * self.sector
        elif kind == "chirper":
          reward += 200.0 * self.sector
      else:
        remaining.append(shot)
    self.shots = remaining
    return reward

  def _resolve_player_contacts(self) -> None:
    contacts = [obj for obj in self.objects if int(obj["row"]) == 9 and int(obj["lane"]) == self.lane]
    for obj in contacts:
      self.objects.remove(obj)
      if obj["kind"] == "rejuvenator" and not bool(obj["red"]):
        self.lives += 1
      else:
        self._lose_ship()

  def _move_sentinel(self) -> float:
    if self.sentinel is None or not bool(self.sentinel["active"]):
      return 0.0
    self.sentinel["column"] = int(self.sentinel["column"]) + 1
    hit = next(
      (
        shot
        for shot in self.shots
        if shot["kind"] == "torpedo"
        and int(shot["row"]) == 0
        and BEAM_COLUMNS[int(shot["lane"])] == int(self.sentinel["column"])
      ),
      None,
    )
    if hit is not None:
      self.shots.remove(hit)
      self.sentinel["active"] = False
      return 500.0 * self.sector + 100.0 * self.lives
    elif int(self.sentinel["column"]) > 9:
      self.sentinel["active"] = False
    return 0.0

  def _lose_ship(self) -> None:
    self.lives -= 1
    self.lane = 2
    self.shots.clear()
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, BEAM_COLUMNS[self.lane], 0] = True
    for obj in self.objects:
      row, column = int(obj["row"]), BEAM_COLUMNS[int(obj["lane"])]
      dangerous = obj["kind"] != "rejuvenator" or bool(obj["red"])
      if dangerous:
        state[row, column, 1] = True
      if obj["kind"] in ("saucer", "chirper") or (obj["kind"] == "rejuvenator" and not bool(obj["red"])):
        state[row, column, 2] = True
      marker = {"saucer": 0, "chirper": 1, "debris": 2, "blocker": 3, "rejuvenator": 4}[str(obj["kind"])]
      state[row, marker, 5] = True
    if self.sentinel is not None and bool(self.sentinel["active"]):
      state[0, min(9, int(self.sentinel["column"])), 2] = True
    for shot in self.shots:
      row, column = int(shot["row"]), BEAM_COLUMNS[int(shot["lane"])]
      state[row, column, 3] = True
      if shot["kind"] == "torpedo":
        state[row, column, 5] = True
    for row, lane in self.previous_objects:
      if 0 <= row < 10:
        state[row, BEAM_COLUMNS[lane], 4] = True
    state[9, : min(10, self.lives), 5] = True
    state[8, : self.torpedoes, 5] = True
    state[7, : self.sector_index + 1, 5] = True
    state[6, : min(10, (15 - self.saucers_destroyed + 1) // 2), 5] = True
    return state
