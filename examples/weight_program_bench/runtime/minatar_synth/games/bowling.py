import numpy as np


PIN_POSITIONS = ((4, 6), (3, 7), (5, 7), (2, 8), (4, 8), (6, 8), (1, 9), (3, 9), (5, 9), (7, 9))


class BowlingEnvironment:
  """Ten-frame steerable bowling with standard strike, spare, and tenth-frame scoring."""

  score_cap = 300.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 500):
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
    self.bowler_row = 4
    self.ball: list[int] | None = None
    self.previous_ball: list[int] | None = None
    self.hook = 0
    self.standing = set(range(10))
    self.frames: list[list[int]] = [[] for _ in range(10)]
    self.frame = 0
    self.confirmed_score = 0
    self.steps = 0
    self.last_action = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if self.ball is None:
      if action == 2:
        self.bowler_row = max(1, self.bowler_row - 1)
      elif action == 4:
        self.bowler_row = min(7, self.bowler_row + 1)
      elif action == 5:
        self.ball = [self.bowler_row, 0]
        self.previous_ball = self.ball.copy()
        self.hook = 0
    else:
      self.previous_ball = self.ball.copy()
      if action == 2:
        self.ball[0] = max(1, self.ball[0] - 1)
        self.hook = max(-1, self.hook - 1)
      elif action == 4:
        self.ball[0] = min(7, self.ball[0] + 1)
        self.hook = min(1, self.hook + 1)
      self.ball[1] += 1
      if self.ball[1] >= 9:
        previous_score = self.confirmed_score
        self._finish_roll()
        self.confirmed_score = self._score_available()
        reward = float(self.confirmed_score - previous_score)
        self.steps += 1
        return reward, self.terminal
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return 0.0, self.terminal

  def _finish_roll(self) -> None:
    knocked = self._knocked_pins()
    self.standing -= knocked
    pinfall = len(knocked)
    frame_rolls = self.frames[self.frame]
    frame_rolls.append(pinfall)
    self.ball = None
    self.previous_ball = None
    self.hook = 0
    if self.frame < 9:
      if pinfall == 10 or len(frame_rolls) == 2:
        self.frame += 1
        self.standing = set(range(10))
      return
    if len(frame_rolls) == 1:
      if pinfall == 10:
        self.standing = set(range(10))
    elif len(frame_rolls) == 2:
      if frame_rolls[0] == 10:
        if pinfall == 10:
          self.standing = set(range(10))
      elif sum(frame_rolls) == 10:
        self.standing = set(range(10))
      else:
        self.terminal = True
    else:
      self.terminal = True

  def _knocked_pins(self) -> set[int]:
    if not self.standing or self.ball is None:
      return set()
    entry_row = self.ball[0]
    if self.standing == set(range(10)) and entry_row == 4 and self.hook != 0:
      return self.standing.copy()
    knocked = {
      index
      for index in self.standing
      if abs(PIN_POSITIONS[index][0] - entry_row) <= 1
    }
    if self.hook < 0:
      knocked |= {index for index in self.standing if PIN_POSITIONS[index][0] < entry_row and abs(PIN_POSITIONS[index][0] - entry_row) <= 2}
    elif self.hook > 0:
      knocked |= {index for index in self.standing if PIN_POSITIONS[index][0] > entry_row and abs(PIN_POSITIONS[index][0] - entry_row) <= 2}
    return knocked

  def _score_available(self) -> int:
    flattened = [roll for frame in self.frames for roll in frame]
    score = 0
    cursor = 0
    for frame_index in range(9):
      frame = self.frames[frame_index]
      if not frame:
        break
      if frame[0] == 10:
        if cursor + 2 >= len(flattened):
          break
        score += 10 + flattened[cursor + 1] + flattened[cursor + 2]
        cursor += 1
      elif len(frame) == 2 and sum(frame) == 10:
        if cursor + 2 >= len(flattened):
          break
        score += 10 + flattened[cursor + 2]
        cursor += 2
      elif len(frame) == 2:
        score += sum(frame)
        cursor += 2
      else:
        break
    tenth = self.frames[9]
    if self.frame == 9 or self.terminal:
      score += sum(tenth)
    return score

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.bowler_row, 0, 0] = True
    state[0, :, 1] = True
    state[9, :, 1] = True
    for index in self.standing:
      state[*PIN_POSITIONS[index], 2] = True
    if self.ball is not None:
      state[*self.ball, 3] = True
    if self.previous_ball is not None:
      state[*self.previous_ball, 4] = True
    state[9, : self.frame + 1, 5] = True
    state[8, : min(10, len(self.frames[self.frame]) + 1), 5] = True
    score_bucket = min(9, self.confirmed_score // 30)
    state[7, : score_bucket + 1, 5] = True
    state[6, self.hook + 1, 5] = True
    return state
