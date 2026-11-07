import numpy as np


DELTAS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}
OPENINGS = {(0, 5), (9, 5), (5, 0), (5, 9)}
BASE_INTERIOR = {(2, 2), (2, 3), (2, 4), (4, 7), (5, 7), (6, 7), (7, 2), (7, 3)}
SPAWN_CANDIDATES = ((1, 1), (1, 8), (4, 4), (7, 7), (8, 1), (3, 8), (6, 2))


def room_walls(room: int) -> set[tuple[int, int]]:
  boundary = {
    (row, column)
    for row in range(10)
    for column in range(10)
    if row in (0, 9) or column in (0, 9)
  } - OPENINGS
  interior = BASE_INTERIOR
  for _ in range(room):
    interior = {(column, 9 - row) for row, column in interior}
  return boundary | interior


class BerzerkEnvironment:
  """Four electrified mazes with robot crossfire and invincible Evil Otto pressure."""

  score_cap = 1_200.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_000):
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
    self.room = 0
    self.lives = 3
    self.steps = 0
    self.last_action = 0
    self.death_this_step = False
    self.terminal = False
    self._start_room([9, 5])

  def _start_room(self, entry: list[int]) -> None:
    self.walls = room_walls(self.room)
    self.player = entry.copy()
    self.previous_player = self.player.copy()
    self.aim = 2
    available = [position for position in SPAWN_CANDIDATES if position not in self.walls and list(position) != self.player]
    self.robots = [list(position) for position in available[:5]]
    self.previous_robots = [robot.copy() for robot in self.robots]
    self.player_shot: list[int] | None = None
    self.robot_shots: list[list[int]] = []
    self.otto: list[int] | None = None
    self.room_steps = 0
    self.clear_bonus_awarded = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.death_this_step = False
    self.previous_player = self.player.copy()
    if action in DELTAS:
      self.aim = action
      if self._move_player(action):
        self.steps += 1
        return 0.0, self.terminal
    elif action == 5 and self.player_shot is None:
      row_delta, column_delta = DELTAS[self.aim]
      self.player_shot = [self.player[0], self.player[1], row_delta, column_delta, 8]

    self.previous_robots = [robot.copy() for robot in self.robots]
    reward = self._move_robots()
    self._move_shots()
    reward += self._resolve_player_shot()
    reward += self._resolve_robot_shots()
    self._spawn_and_move_otto()
    reward += self._resolve_otto()
    if not self.robots and not self.clear_bonus_awarded:
      reward += 50.0
      self.clear_bonus_awarded = True
    self.room_steps += 1
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> bool:
    row_delta, column_delta = DELTAS[action]
    target = [self.player[0] + row_delta, self.player[1] + column_delta]
    if not (0 <= target[0] < 10 and 0 <= target[1] < 10):
      if tuple(self.player) in OPENINGS:
        self.room += 1
        if self.room == 4:
          self.terminal = True
        else:
          opposite = {1: [5, 9], 2: [9, 5], 3: [5, 0], 4: [0, 5]}[action]
          self._start_room(opposite)
        return True
      return False
    if tuple(target) in self.walls:
      self._lose_life()
    elif target in self.robots or target == self.otto:
      self._lose_life()
    else:
      self.player = target
    return False

  def _move_robots(self) -> float:
    if self.room_steps % 2 != 0:
      return 0.0
    reward = 0.0
    for robot in self.robots.copy():
      row_delta = int(np.sign(self.player[0] - robot[0]))
      column_delta = int(np.sign(self.player[1] - robot[1]))
      target = [robot[0] + row_delta, robot[1] + column_delta]
      if tuple(target) in self.walls:
        if robot in self.robots:
          self.robots.remove(robot)
          reward += 50.0
      elif target in self.robots:
        if robot in self.robots:
          self.robots.remove(robot)
          reward += 50.0
        if target in self.robots:
          self.robots.remove(target)
          reward += 50.0
      elif target == self.player:
        self._lose_life()
      elif robot in self.robots:
        robot[:] = target
    if self.room_steps % 8 == 0:
      for robot in self.robots:
        row_delta = int(np.sign(self.player[0] - robot[0]))
        column_delta = int(np.sign(self.player[1] - robot[1]))
        self.robot_shots.append([robot[0], robot[1], row_delta, column_delta, 8])
    return reward

  def _move_shots(self) -> None:
    if self.player_shot is not None:
      self.player_shot[0] += self.player_shot[2]
      self.player_shot[1] += self.player_shot[3]
      self.player_shot[4] -= 1
      if (
        self.player_shot[4] <= 0
        or not (0 <= self.player_shot[0] < 10 and 0 <= self.player_shot[1] < 10)
        or tuple(self.player_shot[:2]) in self.walls
      ):
        self.player_shot = None
    remaining = []
    for shot in self.robot_shots:
      shot[0] += shot[2]
      shot[1] += shot[3]
      shot[4] -= 1
      if shot[4] > 0 and 0 <= shot[0] < 10 and 0 <= shot[1] < 10 and tuple(shot[:2]) not in self.walls:
        remaining.append(shot)
    self.robot_shots = remaining

  def _resolve_player_shot(self) -> float:
    if self.player_shot is None:
      return 0.0
    hit = next((robot for robot in self.robots if robot == self.player_shot[:2]), None)
    if hit is None:
      return 0.0
    self.robots.remove(hit)
    self.player_shot = None
    return 50.0

  def _resolve_robot_shots(self) -> float:
    reward = 0.0
    remaining = []
    for shot in self.robot_shots:
      hit = next((robot for robot in self.robots if robot == shot[:2]), None)
      if hit is not None:
        self.robots.remove(hit)
        reward += 50.0
      elif shot[:2] == self.player:
        self._lose_life()
      else:
        remaining.append(shot)
    self.robot_shots = remaining
    return reward

  def _spawn_and_move_otto(self) -> None:
    if self.otto is None and self.room_steps >= 40:
      self.otto = [9, 5]
    if self.otto is None or self.room_steps % 2 != 0:
      return
    self.otto[0] += int(np.sign(self.player[0] - self.otto[0]))
    self.otto[1] += int(np.sign(self.player[1] - self.otto[1]))

  def _resolve_otto(self) -> float:
    if self.otto is None:
      return 0.0
    reward = 0.0
    hit = next((robot for robot in self.robots if robot == self.otto), None)
    if hit is not None:
      self.robots.remove(hit)
      reward += 50.0
    if self.otto == self.player:
      self._lose_life()
    return reward

  def _lose_life(self) -> None:
    if self.death_this_step:
      return
    self.death_this_step = True
    self.lives -= 1
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.player_shot = None
    self.robot_shots.clear()
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for robot in self.robots:
      state[*robot, 1] = True
      state[*robot, 2] = True
    for shot in self.robot_shots:
      state[shot[0], shot[1], 1] = True
    if self.otto is not None:
      state[*self.otto, 1] = True
    if self.player_shot is not None:
      state[self.player_shot[0], self.player_shot[1], 3] = True
    state[*self.previous_player, 4] = True
    for robot in self.previous_robots:
      state[*robot, 4] = True
    for wall in self.walls:
      state[*wall, 5] = True
    state[9, : self.lives, 5] = True
    state[8, : self.room + 1, 5] = True
    state[7, self.aim, 5] = True
    return state
