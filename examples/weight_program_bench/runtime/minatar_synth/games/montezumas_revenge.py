import numpy as np


ITEM_VALUES = {"key": 100, "sword": 100, "amulet": 200, "jewel": 1_000, "torch": 3_000}
ITEM_LAYOUT = {
  0: (("key", 2), ("sword", 7)),
  1: (("jewel", 4),),
  2: (("amulet", 7), ("sword", 2)),
  3: (("key", 2), ("jewel", 7)),
  4: (("torch", 4), ("sword", 8)),
  5: (("key", 7),),
  6: (("sword", 2),),
}
DOORS = frozenset({(0, 1), (3, 4), (7, 8)})
ROOM_NEIGHBORS = {
  0: {3: 1, 4: 3},
  1: {1: 0, 3: 2, 4: 4},
  2: {1: 1, 4: 5},
  3: {2: 0, 3: 4, 4: 6},
  4: {2: 1, 1: 3, 3: 5, 4: 7},
  5: {2: 2, 1: 4, 4: 8},
  6: {2: 3, 3: 7},
  7: {2: 4, 1: 6, 3: 8},
  8: {2: 5, 1: 7},
}


class MontezumasRevengeEnvironment:
  """Three compact fortress levels with inventory-gated exploration."""

  score_cap = 50_400.0

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
    self.level = 0
    self.lives = 5
    self.score = 0.0
    self.next_bonus_life = 10_000
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed: set[tuple[int, str, int, int]] = set()
    self.open_doors: set[tuple[int, int, int]] = set()
    self.defeated: set[tuple[int, str, int]] = set()
    self.inventory: list[str] = []
    self._start_level()

  def _start_level(self) -> None:
    self.room = 0
    self.player = [8, 1]
    self.previous_player = self.player.copy()
    self.jump_timer = 0
    self.amulet_timer = 0
    self.facing = 1
    self.room_steps = 0
    self.skull_column = 8
    self.skull_direction = -1
    self.spider_column = 1
    self.spider_direction = 1

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    level_before = self.level
    room_before = self.room
    self.transition_reward = 0.0
    self._move_player(action)
    reward = self.transition_reward + self._collect_item()
    if action == 5:
      reward += self._use_sword()
    if self.level == level_before and self.room == room_before:
      self._move_creatures()
      self._resolve_hazards()
    self.score += reward
    while self.score >= self.next_bonus_life:
      self.lives = min(6, self.lives + 1)
      self.next_bonus_life += 10_000
    self.steps += 1
    self.room_steps += 1
    self.jump_timer = max(0, self.jump_timer - 1)
    self.amulet_timer = max(0, self.amulet_timer - 1)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if action == 1:
      self.player[1] -= 1
      self.facing = -1
    elif action == 3:
      self.player[1] += 1
      self.facing = 1
    elif action == 2:
      if self.player[1] == 5 and 2 in ROOM_NEIGHBORS[self.room]:
        self._change_room(2)
        return
      self.jump_timer = 3
      self.player[0] = 6
    elif action == 4 and self.player[1] == 5 and 4 in ROOM_NEIGHBORS[self.room]:
      self._change_room(4)
      return
    if self.jump_timer == 0:
      self.player[0] = 8
    if self.player[1] < 0:
      self._change_room(1)
    elif self.player[1] > 9:
      self._change_room(3)

  def _change_room(self, direction: int) -> None:
    destination = ROOM_NEIGHBORS[self.room].get(direction)
    if destination is None:
      self.player[1] = min(9, max(0, self.player[1]))
      return
    edge = tuple(sorted((self.room, destination)))
    door = (self.level, *edge)
    if edge in DOORS and door not in self.open_doors:
      if "key" not in self.inventory:
        self.player[1] = min(9, max(0, self.player[1]))
        return
      self.inventory.remove("key")
      self.open_doors.add(door)
      self.transition_reward += 300.0
    self.room = destination
    if direction == 1:
      self.player = [8, 9]
    elif direction == 3:
      self.player = [8, 0]
    else:
      self.player = [8, 5]
    self.previous_player = self.player.copy()
    self.room_steps = 0
    self.skull_column = 8
    self.skull_direction = -1
    self.spider_column = 1
    self.spider_direction = 1
    if self.room == 8 and (self.level, 7, 8) in self.open_doors:
      self.level += 1
      if self.level >= 3:
        self.terminal = True
      else:
        self.inventory.clear()
        self._start_level()

  def _collect_item(self) -> float:
    for index, (kind, column) in enumerate(ITEM_LAYOUT.get(self.room, ())):
      identity = (self.level, kind, self.room, index)
      if identity in self.claimed or self.player != [8, column]:
        continue
      if kind != "jewel" and len(self.inventory) >= 4:
        return 0.0
      self.claimed.add(identity)
      if kind != "jewel":
        self.inventory.append(kind)
      if kind == "amulet":
        self.amulet_timer = 20
      return float(ITEM_VALUES[kind])
    return 0.0

  def _use_sword(self) -> float:
    if "sword" not in self.inventory:
      return 0.0
    target = self.player[1] + self.facing
    if self.room in (2, 6):
      identity = (self.level, "skull", self.room)
      if identity not in self.defeated and self.skull_column == target:
        self.defeated.add(identity)
        self.inventory.remove("sword")
        return 2_000.0
    if self.room in (5, 7):
      identity = (self.level, "spider", self.room)
      if identity not in self.defeated and self.spider_column == target:
        self.defeated.add(identity)
        self.inventory.remove("sword")
        return 3_000.0
    return 0.0

  def _move_creatures(self) -> None:
    speed = max(1, 3 - self.level)
    if self.room_steps % speed:
      return
    if self.room in (2, 6) and (self.level, "skull", self.room) not in self.defeated:
      self.skull_column += self.skull_direction
      if self.skull_column in (1, 8):
        self.skull_direction *= -1
    if self.room in (5, 7) and (self.level, "spider", self.room) not in self.defeated:
      self.spider_column += self.spider_direction
      if self.spider_column in (1, 8):
        self.spider_direction *= -1

  def _resolve_hazards(self) -> None:
    if self.amulet_timer > 0:
      return
    creature_hit = (
      self.room in (2, 6)
      and (self.level, "skull", self.room) not in self.defeated
      and self.skull_column == self.player[1]
    ) or (
      self.room in (5, 7)
      and (self.level, "spider", self.room) not in self.defeated
      and self.spider_column == self.player[1]
    )
    fire_hit = self.room in (3, 6) and self.player[1] in (3, 7) and self.jump_timer == 0
    laser_hit = self.room == 4 and self.player[1] == 6 and self.room_steps % 6 < 3
    dark_fall = self.room >= 6 and "torch" not in self.inventory and self.player[1] in (4, 8)
    if creature_hit or fire_hit or laser_hit or dark_fall:
      self._lose_life()

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    room = self.room
    self.player = [8, 1]
    self.previous_player = self.player.copy()
    self.jump_timer = 0
    self.amulet_timer = 0
    self.room_steps = 0
    self.skull_column = 8
    self.skull_direction = -1
    self.spider_column = 1
    self.spider_direction = 1
    self.room = room

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    state[*self.previous_player, 4] = True
    for _kind, column in ITEM_LAYOUT.get(self.room, ()):
      index = list(ITEM_LAYOUT[self.room]).index((_kind, column))
      if (self.level, _kind, self.room, index) not in self.claimed:
        state[8, column, 2] = True
    if self.room in (2, 6) and (self.level, "skull", self.room) not in self.defeated:
      state[8, self.skull_column, 1] = True
    if self.room in (5, 7) and (self.level, "spider", self.room) not in self.defeated:
      state[8, self.spider_column, 1] = True
    state[8, :, 3] = True
    if self.room in (3, 6):
      state[8, (3, 7), 3] = False
    state[:, 5, 3] = True
    item_identities = []
    for level in range(3):
      for room, items in ITEM_LAYOUT.items():
        for index, (kind, _column) in enumerate(items):
          item_identities.append((level, kind, room, index))
    for index, identity in enumerate(item_identities):
      state[index // 10, index % 10, 5] = identity in self.claimed
    door_edges = sorted(DOORS)
    for level in range(3):
      for edge_index, edge in enumerate(door_edges):
        index = 33 + level * 3 + edge_index
        state[index // 10, index % 10, 5] = (level, *edge) in self.open_doors
    creature_identities = []
    for level in range(3):
      for kind, room in (("skull", 2), ("skull", 6), ("spider", 5), ("spider", 7)):
        creature_identities.append((level, kind, room))
    for offset, identity in enumerate(creature_identities):
      index = 42 + offset
      state[index // 10, index % 10, 5] = identity in self.defeated
    inventory_codes = {"key": 1, "sword": 2, "amulet": 3, "torch": 4}
    for slot, kind in enumerate(self.inventory):
      code = inventory_codes[kind]
      for bit in range(3):
        index = 54 + slot * 3 + bit
        state[index // 10, index % 10, 5] = bool(code & (1 << bit))
    for bit in range(2):
      index = 66 + bit
      state[index // 10, index % 10, 5] = bool(self.level & (1 << bit))
    for bit in range(4):
      index = 68 + bit
      state[index // 10, index % 10, 5] = bool(self.room & (1 << bit))
    for bit in range(3):
      index = 72 + bit
      state[index // 10, index % 10, 5] = bool(self.lives & (1 << bit))
    for bit in range(12):
      index = 75 + bit
      state[index // 10, index % 10, 5] = bool(self.steps & (1 << bit))
    for bit in range(2):
      index = 87 + bit
      state[index // 10, index % 10, 5] = bool(self.jump_timer & (1 << bit))
    for bit in range(5):
      index = 89 + bit
      state[index // 10, index % 10, 5] = bool(self.amulet_timer & (1 << bit))
    state[9, 4, 5] = self.facing > 0
    state[9, 5, 5] = self.room >= 6 and "torch" not in self.inventory
    for bit in range(3):
      state[9, 6 + bit, 5] = bool((self.room_steps % 6) & (1 << bit))
    state[9, 9, 5] = self.skull_direction > 0 if self.room in (2, 6) else self.spider_direction > 0
    return state
