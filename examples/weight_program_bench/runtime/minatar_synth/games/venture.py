import numpy as np


ROOM_ENTRANCES = ((2, 2), (2, 7), (7, 2), (7, 7))
DIRECTIONS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class VentureEnvironment:
  """Four-depth Venture expedition with hall and chamber scales."""

  score_cap = 20_000.0

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
    self.depth = 1
    self.lives = 4
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.facing = 2
    self.terminal = False
    self._start_hall()

  def _start_hall(self) -> None:
    self.mode = "hall"
    self.player = [9, 5]
    self.completed_rooms: set[int] = set()
    self.hallmonsters = [[0, 0], [0, 9]]
    self.current_room: int | None = None
    self.bullets: list[list[int]] = []

  def _enter_room(self, room: int) -> None:
    self.mode = "room"
    self.current_room = room
    self.player = [8, 5]
    self.has_treasure = False
    self.room_clock = 0
    self.treasure = [2, 5]
    self.monsters = [
      {"id": 0, "row": 2, "column": 2},
      {"id": 1, "row": 2, "column": 7},
      {"id": 2, "row": 5, "column": 2},
    ]
    self.corpses: list[dict[str, int]] = []
    self.intruder: list[int] | None = None
    self.bullets.clear()

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    reward = 0.0
    if action in DIRECTIONS:
      self.facing = action
      self._move_player(action)
    elif action == 5 and self.mode == "room" and self.current_room != 3:
      shot_direction = self._shot_direction()
      self.facing = shot_direction
      row_delta, column_delta = DIRECTIONS[shot_direction]
      self.bullets.append([self.player[0], self.player[1], row_delta, column_delta])
    if self.mode == "hall":
      self._move_hallmonsters()
      self._resolve_hall()
    else:
      self.room_clock += 1
      self._move_bullets()
      reward += self._resolve_shots()
      self._move_room_hazards()
      reward += self._resolve_room()
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _shot_direction(self) -> int:
    aligned = []
    for monster in self.monsters:
      if monster["row"] == self.player[0]:
        direction = 1 if monster["column"] < self.player[1] else 3
        aligned.append((abs(monster["column"] - self.player[1]), direction))
      elif monster["column"] == self.player[1]:
        direction = 2 if monster["row"] < self.player[0] else 4
        aligned.append((abs(monster["row"] - self.player[0]), direction))
    return min(aligned)[1] if aligned else self.facing

  def _move_player(self, action: int) -> None:
    row_delta, column_delta = DIRECTIONS[action]
    destination = [self.player[0] + row_delta, self.player[1] + column_delta]
    if not (0 <= destination[0] < 10 and 0 <= destination[1] < 10):
      return
    if self.mode == "room" and (destination[0] in (0, 9) or destination[1] in (0, 9)):
      if destination == [9, 5]:
        self._leave_room()
      return
    self.player = destination

  def _leave_room(self) -> None:
    room = self.current_room
    assert room is not None
    if self.has_treasure:
      self.completed_rooms.add(room)
    self.mode = "hall"
    self.player = list(ROOM_ENTRANCES[room])
    self.current_room = None
    self.bullets.clear()
    if len(self.completed_rooms) == 4:
      self.depth += 1
      if self.depth == 5:
        self.terminal = True
      else:
        self._start_hall()

  def _move_hallmonsters(self) -> None:
    if self.steps % max(2, 6 - self.depth):
      return
    for monster in self.hallmonsters:
      if abs(self.player[0] - monster[0]) >= abs(self.player[1] - monster[1]):
        monster[0] += int(np.sign(self.player[0] - monster[0]))
      else:
        monster[1] += int(np.sign(self.player[1] - monster[1]))

  def _resolve_hall(self) -> None:
    if self.player in self.hallmonsters:
      self._lose_life()
      return
    for room, entrance in enumerate(ROOM_ENTRANCES):
      if tuple(self.player) == entrance and room not in self.completed_rooms:
        self._enter_room(room)
        return

  def _move_bullets(self) -> None:
    for bullet in self.bullets:
      bullet[0] += bullet[2]
      bullet[1] += bullet[3]
    self.bullets = [bullet for bullet in self.bullets if 0 < bullet[0] < 9 and 0 < bullet[1] < 9]

  def _resolve_shots(self) -> float:
    hit_bullets: set[int] = set()
    hit_monsters: set[int] = set()
    reward = 0.0
    for bullet_index, bullet in enumerate(self.bullets):
      for monster in self.monsters:
        if bullet[:2] == [monster["row"], monster["column"]]:
          hit_bullets.add(bullet_index)
          hit_monsters.add(monster["id"])
          self.corpses.append({"row": monster["row"], "column": monster["column"], "timer": 6})
          if self.has_treasure:
            reward += 100 * self.depth
          break
    self.bullets = [bullet for index, bullet in enumerate(self.bullets) if index not in hit_bullets]
    self.monsters = [monster for monster in self.monsters if monster["id"] not in hit_monsters]
    return float(reward)

  def _move_room_hazards(self) -> None:
    if self.current_room == 3:
      return
    interval = max(2, 6 - self.depth)
    if self.room_clock % interval == 0:
      for monster in self.monsters:
        monster["row"] += int(np.sign(self.player[0] - monster["row"]))
        monster["column"] += int(np.sign(self.player[1] - monster["column"]))
    for corpse in self.corpses:
      corpse["timer"] -= 1
    self.corpses = [corpse for corpse in self.corpses if corpse["timer"] > 0]
    if self.room_clock >= 60 and self.intruder is None:
      self.intruder = [1, 5]
    if self.intruder is not None:
      self.intruder[0] += int(np.sign(self.player[0] - self.intruder[0]))
      self.intruder[1] += int(np.sign(self.player[1] - self.intruder[1]))

  def _wall_cells(self) -> set[tuple[int, int]]:
    phase = (self.room_clock // 3) % 4
    return {(3 + phase, column) for column in range(2, 8)} if self.current_room == 3 else set()

  def _resolve_room(self) -> float:
    reward = 0.0
    if self.player == self.treasure and not self.has_treasure:
      self.has_treasure = True
      reward += 200 * self.depth
    lethal = any([monster["row"], monster["column"]] == self.player for monster in self.monsters)
    lethal |= any([corpse["row"], corpse["column"]] == self.player for corpse in self.corpses)
    lethal |= self.intruder == self.player
    lethal |= tuple(self.player) in self._wall_cells()
    if lethal:
      self._lose_life()
    return float(reward)

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives == 0:
      self.terminal = True
      return
    if self.mode == "hall":
      self.player = [9, 5]
      self.hallmonsters = [[0, 0], [0, 9]]
    else:
      room = self.current_room
      assert room is not None
      self._enter_room(room)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.player[0], self.player[1], 0] = True
    if self.mode == "hall":
      for monster in self.hallmonsters:
        state[monster[0], monster[1], 1] = True
      for room, entrance in enumerate(ROOM_ENTRANCES):
        if room not in self.completed_rooms:
          state[entrance[0], entrance[1], 2] = True
    else:
      for monster in self.monsters:
        state[monster["row"], monster["column"], 1] = True
      for corpse in self.corpses:
        state[corpse["row"], corpse["column"], 4] = True
      if self.intruder is not None:
        state[self.intruder[0], self.intruder[1], 1] = True
      if not self.has_treasure:
        state[self.treasure[0], self.treasure[1], 2] = True
      for bullet in self.bullets:
        state[bullet[0], bullet[1], 3] = True
      for row, column in self._wall_cells():
        state[row, column, 4] = True
    values = ((self.depth, 3), (self.lives, 3), (len(self.completed_rooms), 3), (self.room_clock if self.mode == "room" else 0, 6))
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[2, 0, 5] = self.mode == "room"
    state[2, 1, 5] = getattr(self, "has_treasure", False)
    return state
