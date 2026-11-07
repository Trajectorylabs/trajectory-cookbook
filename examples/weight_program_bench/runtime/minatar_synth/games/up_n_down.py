import numpy as np


LANE_COLUMNS = (2, 5, 8)
FLAG_SCORE = 75
PRIZE_SCORES = (50, 65, 70, 75)
OPPONENT_SCORES = (100, 125, 150, 175)
COURSE_LENGTH = 80


def road_lanes(position: int) -> frozenset[int]:
  segment = position % COURSE_LENGTH
  if segment < 8:
    return frozenset((1,))
  if segment < 16:
    return frozenset((0, 1))
  if segment < 24:
    return frozenset((0, 1, 2))
  if segment < 36:
    return frozenset((0, 2))
  if segment < 44:
    return frozenset((0, 1, 2))
  if segment < 56:
    return frozenset((1, 2))
  if segment < 64:
    return frozenset((0, 1, 2))
  return frozenset((1,))


class UpNDownEnvironment:
  """Nine finite roads of branching, jumping Atari 2600 racing."""

  score_cap = 17_640.0

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
    self.round = 0
    self.lives = 5
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_round()

  def _start_round(self) -> None:
    self.position = 0
    self.lane = 1
    self.direction = 1
    self.speed = 1
    self.airborne = 0
    self.flags = {
      (position, lane): identifier
      for identifier, (position, lane) in enumerate(
        ((10, 1), (18, 0), (22, 2), (30, 0), (34, 2), (48, 1), (58, 2), (70, 1))
      )
    }
    self.prizes = {
      (position, lane): identifier
      for identifier, (position, lane) in enumerate(((13, 0), (27, 2), (51, 1), (62, 0)))
    }
    opponent_positions = ((15, 0), (20, 2), (29, 0), (33, 2), (46, 1), (54, 2), (61, 0), (73, 1))
    self.opponents = [
      {
        "id": identifier,
        "kind": identifier // 2,
        "position": position,
        "lane": lane,
        "direction": -1 if identifier % 2 else 1,
      }
      for identifier, (position, lane) in enumerate(opponent_positions)
    ]
    self.collected_flags: set[int] = set()

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if action == 1:
      self._change_lane(-1)
    elif action == 2:
      self.direction = 1
      self.speed = 2
    elif action == 3:
      self._change_lane(1)
    elif action == 4:
      self.direction = -1
      self.speed = 1
    elif action == 5 and self.direction > 0 and self.airborne == 0:
      self.airborne = 3
    reward = self._advance()
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _change_lane(self, delta: int) -> None:
    destination = self.lane + delta
    if 0 <= destination < 3 and destination in road_lanes(self.position):
      self.lane = destination

  def _advance(self) -> float:
    reward = 0.0
    previous_position = self.position
    self.position = (self.position + self.direction * self.speed) % COURSE_LENGTH
    if self.airborne > 0:
      self.airborne -= 1
    if self.lane not in road_lanes(self.position) and self.airborne == 0:
      self._lose_life()
      return reward
    flag = self.flags.pop((self.position, self.lane), None)
    if flag is not None:
      self.collected_flags.add(flag)
      reward += FLAG_SCORE
    prize = self.prizes.pop((self.position, self.lane), None)
    if prize is not None:
      reward += PRIZE_SCORES[prize]
    for opponent in self.opponents:
      opponent["position"] = (opponent["position"] + opponent["direction"]) % COURSE_LENGTH
    collisions = [
      opponent
      for opponent in self.opponents
      if opponent["position"] == self.position and opponent["lane"] == self.lane
    ]
    if collisions:
      if self.airborne > 0:
        for opponent in collisions:
          reward += OPPONENT_SCORES[opponent["kind"]]
        collided_ids = {opponent["id"] for opponent in collisions}
        self.opponents = [opponent for opponent in self.opponents if opponent["id"] not in collided_ids]
      else:
        self._lose_life()
    if len(self.collected_flags) == 8:
      self.round += 1
      if self.round == 9:
        self.terminal = True
      else:
        self._start_round()
    elif previous_position > self.position and self.direction > 0:
      self.speed = 1
    return float(reward)

  def _lose_life(self) -> None:
    self.lives -= 1
    self.position = 0
    self.lane = 1
    self.direction = 1
    self.speed = 1
    self.airborne = 0
    if self.lives == 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, LANE_COLUMNS[self.lane], 0] = True
    for row in range(10):
      world_position = (self.position + 8 - row) % COURSE_LENGTH
      for lane in road_lanes(world_position):
        state[row, LANE_COLUMNS[lane], 4] = True
    for (position, lane), _identifier in self.flags.items():
      row = 8 - ((position - self.position) % COURSE_LENGTH)
      if 0 <= row < 10:
        state[row, LANE_COLUMNS[lane], 2] = True
    for (position, lane), _identifier in self.prizes.items():
      row = 8 - ((position - self.position) % COURSE_LENGTH)
      if 0 <= row < 10:
        state[row, LANE_COLUMNS[lane], 3] = True
    for opponent in self.opponents:
      row = 8 - ((opponent["position"] - self.position) % COURSE_LENGTH)
      if 0 <= row < 10:
        state[row, LANE_COLUMNS[opponent["lane"]], 1] = True
    values = (
      (self.round, 4),
      (self.position, 7),
      (self.lane, 2),
      (self.lives, 3),
      (self.speed, 2),
      (self.airborne, 2),
      (len(self.collected_flags), 4),
      (len(self.opponents), 4),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[3, 0, 5] = self.direction > 0
    return state
