import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 8
BASES = ((50, 90), (90, 50), (50, 10), (10, 50))
LEG_LENGTH = math.dist(BASES[0], BASES[1])
RUNNER_SPEED = 0.6
BALL_SPEED = 0.3


class BaserunnerDashEnvironment:
  score_cap = HORIZON * FRAMES_PER_STEP // math.ceil(LEG_LENGTH / RUNNER_SPEED)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.runner_base = 1
    self.progress = 0.0
    self.ball = BASES[0]
    self.ball_base = 0
    self.ball_target = 2
    self.thrown = True

  @property
  def runner_target(self) -> int:
    return (self.runner_base + 1) % len(BASES)

  @property
  def runner_position(self) -> tuple[float, float]:
    x, y = BASES[self.runner_base]
    target_x, target_y = BASES[self.runner_target]
    fraction = self.progress / LEG_LENGTH
    return x + (target_x - x) * fraction, y + (target_y - y) * fraction

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_runner()
      self._advance_ball()
      runner_x, runner_y = self.runner_position
      ball_x, ball_y = self.ball
      if self.progress > 0 and not self.thrown and abs(runner_x - ball_x) < 6 and abs(runner_y - ball_y) < 6:
        self.terminal = True
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_runner(self) -> int:
    if not self.pressed:
      self.progress = max(0, self.progress - RUNNER_SPEED)
      return 0
    self.progress += RUNNER_SPEED
    if self.progress < LEG_LENGTH:
      return 0
    self.runner_base = self.runner_target
    self.progress = 0.0
    return 1

  def _advance_ball(self) -> None:
    x, y = self.ball
    target = BASES[self.ball_target]
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    speed = BALL_SPEED * (3 if self.thrown else 1)
    if distance <= speed:
      self.ball = target
      self.ball_base = self.ball_target
      self.thrown = False
      self.ball_target = self.runner_target if self.ball_base == self.runner_base else self.runner_base
      return
    self.ball = x + dx * speed / distance, y + dy * speed / distance
    if not self.thrown and self.random.random() < 0.05:
      self.thrown = True
      if self.ball_base not in (self.runner_base, self.runner_target):
        self.ball_target = self.runner_target

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for x, y in BASES:
      state[y // 10, x // 10, 0] = True
    runner_x, runner_y = self.runner_position
    state[int(runner_y) // 10, int(runner_x) // 10, 1] = True
    ball_x, ball_y = self.ball
    state[int(ball_y) // 10, int(ball_x) // 10, 2 if self.thrown else 3] = True
    target_x, target_y = BASES[self.ball_target]
    state[target_y // 10, target_x // 10, 4] = True
    for index, channel in ((self.runner_base, 4), (self.runner_target, 5)):
      state[0, index, channel] = True
    for bit in range(9):
      state[9, bit, 5] = bool(self.steps & (1 << bit))
    state[9, 9, 5] = self.pressed
    return state
