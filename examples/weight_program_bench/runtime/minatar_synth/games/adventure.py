import numpy as np


class AdventureEnvironment:
  """Compact Adventure: rooms, keys, carried objects, dragon, bat, and chalice return."""

  score_cap = 1.0

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_000):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = horizon
    self.random = np.random.RandomState()
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random.seed(seed)

  def reset(self) -> None:
    self.room = 0
    self.player = [8, 5]
    self.held = None
    self.items = {
      "gold_key": [0, 5, 7], "black_key": [2, 7, 7], "white_key": [4, 2, 2],
      "sword": [1, 7, 2], "chalice": [5, 2, 7], "bridge": [3, 6, 2],
      "magnet": [5, 7, 2],
    }
    self.dragons = [[2, 5, 5], [3, 3, 7], [5, 6, 6]]
    self.previous_dragons = [dragon.copy() for dragon in self.dragons]
    self.bat = [1, 2, 8]
    self.previous_bat = self.bat.copy()
    self.castles = {
      0: {"key": "gold_key", "open": False},
      2: {"key": "black_key", "open": False},
      4: {"key": "white_key", "open": False},
    }
    self.dragon_alive = [True, True, True]
    self.last_action = 0
    self.steps = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    dropped = False
    if action in (1, 2, 3, 4):
      dy, dx = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}[action]
      target = [min(9, max(0, self.player[0] + dy)), min(9, max(0, self.player[1] + dx))]
      if self._can_move(target):
        self.player = target
        self._change_room()
    elif action == 5 and self.held is not None:
      self.items[self.held] = [self.room, *self.player]
      self.held = None
      dropped = True

    if not dropped:
      self._collect_item()
    self._use_magnet()
    self.previous_dragons = [dragon.copy() for dragon in self.dragons]
    self.previous_bat = self.bat.copy()
    self._move_dragons()
    self._move_bat()
    castle = self.castles.get(self.room)
    if castle and self.player[0] <= 1 and self.held == castle["key"]:
      castle["open"] = True
    if self.room == 0 and self.player[0] <= 1 and self.held == "chalice" and self.castles[0]["open"]:
      self.terminal = True
      return 1.0, True
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return 0.0, self.terminal

  def _can_move(self, target: list[int]) -> bool:
    if self.room != 3 or {self.player[0], target[0]} != {4, 5}:
      return True
    bridge = self.items.get("bridge")
    bridge_column = bridge[2] if bridge is not None and bridge[0] == 3 else None
    return self.held == "bridge" or bridge_column == target[1]

  def _change_room(self) -> None:
    if self.player[1] == 0 and self.room % 2 == 1:
      self.room -= 1
      self.player[1] = 9
    elif self.player[1] == 9 and self.room % 2 == 0:
      self.room += 1
      self.player[1] = 0
    elif self.player[0] == 0 and self.room >= 2:
      self.room -= 2
      self.player[0] = 9
    elif self.player[0] == 9 and self.room < 4:
      self.room += 2
      self.player[0] = 0

  def _collect_item(self) -> None:
    for name, location in list(self.items.items()):
      if location == [self.room, *self.player] and self.held is None:
        self.held = name
        del self.items[name]
    castle = self.castles.get(self.room)
    if castle and self.held == castle["key"] and self.player[0] <= 1:
      castle["open"] = True

  def _move_dragons(self) -> None:
    for index, dragon in enumerate(self.dragons):
      if not self.dragon_alive[index] or dragon[0] != self.room:
        continue
      _, row, column = dragon
      row += int(np.sign(self.player[0] - row))
      column += int(np.sign(self.player[1] - column))
      dragon[1:] = [row, column]
      if [row, column] == self.player:
        if self.held == "sword":
          self.dragon_alive[index] = False
        else:
          self.terminal = True

  def _use_magnet(self) -> None:
    if self.held != "magnet":
      return
    candidates = [(name, location) for name, location in self.items.items() if location[0] == self.room]
    if not candidates:
      return
    _, location = min(candidates, key=lambda item: abs(item[1][1] - self.player[0]) + abs(item[1][2] - self.player[1]))
    location[1] += int(np.sign(self.player[0] - location[1]))
    location[2] += int(np.sign(self.player[1] - location[2]))

  def _move_bat(self) -> None:
    if self.random.rand() >= 0.15:
      return
    self.bat[0] = int(self.random.randint(6))
    self.bat[1] = int(self.random.randint(10))
    self.bat[2] = int(self.random.randint(10))
    if self.bat == [self.room, *self.player] and self.held is not None:
      name = self.held
      self.held = None
      self.items[name] = self.bat.copy()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for name, (room, row, column) in self.items.items():
      if room == self.room:
        if name == "chalice":
          state[row, column, 2] = True
        else:
          state[row, column, 3] = True
          marker = {
            "gold_key": 0,
            "black_key": 1,
            "white_key": 2,
            "sword": 3,
            "bridge": 4,
            "magnet": 5,
          }[name]
          state[row, marker, 5] = True
    for alive, dragon in zip(self.dragon_alive, self.dragons, strict=True):
      if alive and dragon[0] == self.room:
        state[dragon[1], dragon[2], 1] = True
    for previous, alive in zip(self.previous_dragons, self.dragon_alive, strict=True):
      if alive and previous[0] == self.room:
        state[previous[1], previous[2], 4] = True
    if self.bat[0] == self.room:
      state[self.bat[1], self.bat[2], 1] = True
    if self.previous_bat[0] == self.room:
      state[self.previous_bat[1], self.previous_bat[2], 4] = True
    if self.room == 3:
      state[4, :, 1] = True
      bridge = self.items.get("bridge")
      if self.held == "bridge":
        state[4, self.player[1], 1] = False
      elif bridge is not None and bridge[0] == 3:
        state[4, bridge[2], 1] = False
    if self.room in self.castles:
      state[0, 4:7, 1] = not self.castles[self.room]["open"]
    if self.held is not None:
      held_marker = {
        "gold_key": 0,
        "black_key": 1,
        "white_key": 2,
        "sword": 3,
        "chalice": 4,
        "bridge": 5,
        "magnet": 6,
      }[self.held]
      state[8, held_marker, 5] = True
    state[9, self.room, 5] = True
    for castle_room, castle in self.castles.items():
      if castle["open"]:
        state[7, castle_room, 5] = True
    return state
