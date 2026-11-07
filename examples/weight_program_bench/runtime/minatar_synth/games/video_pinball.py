import numpy as np


BUMPER_POSITIONS = ((3, 3), (3, 6), (5, 3), (5, 6))
DROP_TARGETS = ((2, 2), (2, 4), (2, 6), (2, 7))
SPINNER = (4, 5)
ATARI_ROLLOVER = (1, 8)
LEFT_ROLLOVER = (1, 1)
SPECIAL_TARGET = (6, 5)


class VideoPinballEnvironment:
  """Finite game-one table with flippers, nudge, tilt, and scoring devices."""

  score_cap = 1_064_000.0

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
    self.balls = 3
    self.extra_ball_available = True
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.bumper_value = 1
    self.terminal = False
    self._serve_ball()

  def _serve_ball(self) -> None:
    self.ball = [8, 8]
    self.velocity = [0, 0]
    self.in_plunger = True
    self.plunger = 0
    self.tilted = False
    self.nudges = 0
    self.atari_rollovers = 0
    self.left_rollovers = 0
    self.multiplier = 1
    self.drop_targets: set[int] = set()
    self.special_timer = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    left_flipper = action in (1, 2)
    right_flipper = action in (2, 3)
    if self.in_plunger:
      if action == 4:
        self.plunger = min(3, self.plunger + 1)
      elif action == 5:
        self.in_plunger = False
        self.velocity = [-1, -1 if self.plunger <= 1 else 0]
      reward = 0.0
    else:
      if action == 5:
        self.nudges += 1
        self.velocity[0] = -1
        if self.nudges >= 4:
          self.tilted = True
      else:
        self.nudges = max(0, self.nudges - 1)
      reward = self._advance_ball(left_flipper, right_flipper)
    if self.special_timer > 0:
      self.special_timer -= 1
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _advance_ball(self, left_flipper: bool, right_flipper: bool) -> float:
    self.ball[0] += self.velocity[0]
    self.ball[1] += self.velocity[1]
    if self.ball[1] <= 0:
      self.ball[1] = 1
      self.velocity[1] = 1
    elif self.ball[1] >= 9:
      self.ball[1] = 8
      self.velocity[1] = -1
    if self.ball[0] <= 0:
      self.ball[0] = 1
      self.velocity[0] = 1
    if self.ball[0] >= 8:
      if left_flipper and 1 <= self.ball[1] <= 4:
        self.ball[0] = 7
        self.velocity = [-1, 1]
      elif right_flipper and 5 <= self.ball[1] <= 8:
        self.ball[0] = 7
        self.velocity = [-1, -1]
      elif self.ball[0] >= 9:
        return self._drain()
    if self.tilted:
      self.velocity[0] = 1
      return 0.0
    return self._score_device()

  def _score_device(self) -> float:
    position = tuple(self.ball)
    reward = 0.0
    if position in BUMPER_POSITIONS:
      reward = 100.0 * self.bumper_value
      self.velocity[0] *= -1
      self.velocity[1] = -self.velocity[1] or 1
    elif position in DROP_TARGETS:
      target = DROP_TARGETS.index(position)
      if target not in self.drop_targets:
        self.drop_targets.add(target)
        reward = 100.0
        if len(self.drop_targets) == 4:
          self.drop_targets.clear()
          self.bumper_value = min(5, self.bumper_value + 1)
          self.special_timer = 4
    elif position == SPINNER:
      reward = 1.0
    elif position == ATARI_ROLLOVER:
      reward = 100.0
      self.atari_rollovers = min(4, self.atari_rollovers + 1)
      self.multiplier = min(4, self.multiplier + 1)
      if self.atari_rollovers == 4 and self.extra_ball_available:
        self.balls += 1
        self.extra_ball_available = False
    elif position == LEFT_ROLLOVER:
      reward = 100.0
      self.left_rollovers = min(4, self.left_rollovers + 1)
    elif position == SPECIAL_TARGET and self.special_timer > 0:
      reward = 100.0
      self.special_timer = 0
    return reward

  def _drain(self) -> float:
    reward = float(self.left_rollovers * 1_000 * self.multiplier)
    self.balls -= 1
    if self.tilted:
      self.balls = min(self.balls, 2)
    if self.balls <= 0:
      self.terminal = True
    else:
      self._serve_ball()
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.ball[0], self.ball[1], 0] = True
    for row, column in BUMPER_POSITIONS:
      state[row, column, 1] = True
    for index, (row, column) in enumerate(DROP_TARGETS):
      if index not in self.drop_targets:
        state[row, column, 2] = True
    for row, column in (SPINNER, ATARI_ROLLOVER, LEFT_ROLLOVER, SPECIAL_TARGET):
      state[row, column, 2] = True
    state[8, 3, 3] = True
    state[8, 6, 3] = True
    state[max(0, min(9, self.ball[0] - self.velocity[0])), max(0, min(9, self.ball[1] - self.velocity[1])), 4] = True
    values = (
      (self.balls, 3),
      (self.bumper_value, 3),
      (self.atari_rollovers, 3),
      (self.left_rollovers, 3),
      (self.multiplier, 3),
      (self.plunger, 2),
      (self.nudges, 3),
      (self.special_timer, 3),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[3, 0, 5] = self.in_plunger
    state[3, 1, 5] = self.tilted
    state[3, 2, 5] = self.extra_ball_available
    return state
