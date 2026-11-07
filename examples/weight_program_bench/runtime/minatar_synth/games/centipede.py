import numpy as np


INITIAL_MUSHROOMS = (
  (1, 1), (1, 5), (1, 8), (2, 3), (2, 7),
  (3, 1), (3, 5), (3, 9), (4, 2), (4, 6),
  (5, 0), (5, 4), (5, 8), (6, 2), (6, 7),
)


class CentipedeEnvironment:
  """Four-wave mushroom-field shooter with splitting chains and specialist insects."""

  score_cap = 9_528.0

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
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.shot: list[int] | None = None
    self.mushrooms = {position: {"health": 3, "poisoned": False} for position in INITIAL_MUSHROOMS}
    self.lives = 3
    self.wave = 0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_wave()

  def _start_wave(self) -> None:
    head_count = self.wave + 1
    main_length = 10 - head_count
    self.chains = [self._make_chain(main_length, 0, 1)]
    for index in range(head_count - 1):
      self.chains.append(self._make_chain(1, index + 1, -1 if index % 2 else 1))
    self.spider: list[int] | None = [8, 9]
    self.flea: dict[str, object] | None = {"position": [0, int(self.random.randint(10))], "health": 2}
    self.scorpion: list[int] | None = [3, 0] if self.wave >= 3 else None
    self.previous_insects: list[list[int]] = []
    self.wave_steps = 0

  @staticmethod
  def _make_chain(length: int, row: int, direction: int) -> dict[str, object]:
    start = 0 if direction > 0 else 9
    segments = [[row, (start - direction * index) % 10] for index in range(length)]
    return {"segments": segments, "direction": direction, "diving": False}

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    if action in (1, 2, 3, 4):
      row_delta, column_delta = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}[action]
      target = [self.player[0] + row_delta, self.player[1] + column_delta]
      if 7 <= target[0] <= 9 and 0 <= target[1] < 10:
        self.player = target
    elif action == 5 and self.shot is None:
      self.shot = [self.player[0] - 1, self.player[1]]

    self.previous_insects = self._insect_positions()
    self._move_centipedes()
    self._move_specials()
    if self.shot is not None:
      self.shot[0] -= 1
      if self.shot[0] < 0:
        self.shot = None
    reward = self._resolve_shot()
    reward += self._resolve_contacts()
    self.wave_steps += 1
    self.steps += 1
    if not self.chains:
      self.wave += 1
      if self.wave == 4:
        self.terminal = True
      else:
        self._start_wave()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_centipedes(self) -> None:
    for chain in self.chains:
      segments = chain["segments"]
      previous = [segment.copy() for segment in segments]
      head = segments[0]
      if bool(chain["diving"]):
        head[0] = min(9, head[0] + 1)
        if head[0] >= 8:
          chain["diving"] = False
          chain["direction"] = -int(chain["direction"])
      else:
        target_column = head[1] + int(chain["direction"])
        target = (head[0], target_column)
        if target_column < 0 or target_column > 9 or target in self.mushrooms:
          mushroom = self.mushrooms.get(target)
          if mushroom is not None and bool(mushroom["poisoned"]):
            chain["diving"] = True
          head[0] = min(9, head[0] + 1)
          chain["direction"] = -int(chain["direction"])
        else:
          head[1] = target_column
      for index in range(1, len(segments)):
        segments[index][:] = previous[index - 1]

  def _move_specials(self) -> None:
    if self.spider is not None:
      self.spider[1] += -1 if self.wave_steps % 4 < 2 else 1
      self.spider[0] = int(np.clip(self.spider[0] + (-1 if self.wave_steps % 3 == 0 else 1), 7, 9))
      if self.spider[1] < 0 or self.spider[1] > 9:
        self.spider[1] %= 10
    if self.flea is not None:
      position = self.flea["position"]
      position[0] += 2 if int(self.flea["health"]) == 1 else 1
      if position[0] in (2, 4, 6) and tuple(position) not in self.mushrooms:
        self.mushrooms[tuple(position)] = {"health": 3, "poisoned": False}
      if position[0] > 9:
        self.flea = None
    if self.scorpion is not None:
      self.scorpion[1] += 1
      position = tuple(self.scorpion)
      if position in self.mushrooms:
        self.mushrooms[position]["poisoned"] = True
      if self.scorpion[1] > 9:
        self.scorpion = None

  def _resolve_shot(self) -> float:
    if self.shot is None:
      return 0.0
    for chain in self.chains.copy():
      segments = chain["segments"]
      hit_index = next((index for index, segment in enumerate(segments) if segment == self.shot), None)
      if hit_index is None:
        continue
      position = tuple(segments[hit_index])
      reward = 100.0 if hit_index == 0 else 10.0
      before = segments[:hit_index]
      after = segments[hit_index + 1:]
      self.chains.remove(chain)
      if before:
        self.chains.append({"segments": before, "direction": chain["direction"], "diving": chain["diving"]})
      if after:
        self.chains.append({"segments": after, "direction": -int(chain["direction"]), "diving": False})
      self.mushrooms.setdefault(position, {"health": 3, "poisoned": False})
      self.shot = None
      return reward
    if self.spider == self.shot:
      distance = abs(self.spider[0] - self.player[0]) + abs(self.spider[1] - self.player[1])
      self.spider = None
      self.shot = None
      return 900.0 if distance <= 2 else 600.0 if distance <= 4 else 300.0
    if self.flea is not None and self.flea["position"] == self.shot:
      self.flea["health"] = int(self.flea["health"]) - 1
      self.shot = None
      if int(self.flea["health"]) == 0:
        self.flea = None
        return 200.0
      return 0.0
    if self.scorpion == self.shot:
      self.scorpion = None
      self.shot = None
      for mushroom in self.mushrooms.values():
        mushroom["poisoned"] = False
      return 1_000.0
    shot_position = tuple(self.shot)
    mushroom = self.mushrooms.get(shot_position)
    if mushroom is not None:
      mushroom["health"] = int(mushroom["health"]) - 1
      self.shot = None
      if int(mushroom["health"]) == 0:
        del self.mushrooms[shot_position]
        return 1.0
    return 0.0

  def _resolve_contacts(self) -> float:
    hazards = self._insect_positions()
    if self.player not in hazards:
      return 0.0
    bonus = float((len(self.mushrooms) // 2) * 5)
    self.lives -= 1
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.shot = None
    for mushroom in self.mushrooms.values():
      mushroom["poisoned"] = False
      mushroom["health"] = 3
    if self.lives <= 0:
      self.terminal = True
    return bonus

  def _insect_positions(self) -> list[list[int]]:
    positions = [segment.copy() for chain in self.chains for segment in chain["segments"]]
    if self.spider is not None:
      positions.append(self.spider.copy())
    if self.flea is not None:
      positions.append(list(self.flea["position"]))
    if self.scorpion is not None:
      positions.append(self.scorpion.copy())
    return positions

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for chain in self.chains:
      for index, segment in enumerate(chain["segments"]):
        state[*segment, 1] = True
        state[*segment, 2] = True
        if index == 0:
          state[segment[0], segment[1], 5] = True
    for position in (self.spider, None if self.flea is None else self.flea["position"], self.scorpion):
      if position is not None:
        state[*position, 1] = True
        state[*position, 2] = True
    for position, mushroom in self.mushrooms.items():
      state[*position, 2] = True
      state[*position, 5] = True
      if bool(mushroom["poisoned"]):
        state[*position, 1] = True
    if self.shot is not None:
      state[*self.shot, 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_insects:
      if 0 <= position[0] < 10 and 0 <= position[1] < 10:
        state[*position, 4] = True
    state[9, : self.lives, 5] = True
    state[8, : self.wave + 1, 5] = True
    return state
