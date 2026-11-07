from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np


@dataclass(eq=False)
class Runner:
  x: float
  y: float
  vx: float = 0
  vy: float = 0
  grounded: bool = False
  jumped: bool = False
  below: "Runner | None" = None
  above: "Runner | None" = None


class SurvivorEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.runners, self.downed = [], []
    self.steps = 0
    self.held = self.terminal = False
    self._wave()

  def _runner(self):
    return Runner(self.random.uniform(10, 40), self.random.uniform(-9, 9))

  def _wave(self):
    self.runners.extend(self._runner() for _ in range(9 - len(self.runners)))
    self.radius = self.random.uniform(5, 25)
    self.barrel_x = 120 + self.radius
    self.barrel_speed = self.random.uniform(1, 2) / sqrt(1 + self.radius * 0.3)
    self.next_wave = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    if self.next_wave:
      self._wave()
    self.barrel_x -= self.barrel_speed
    jumped = []
    replacements = 0
    for runner in tuple(self.runners):
      if runner.grounded and runner.below is None:
        self._stack(runner)
      if tap and (runner.grounded or (runner.below is not None and runner.below.jumped)):
        self._detach(runner)
        runner.vx, runner.vy, runner.grounded = 0, -1.5, False
        jumped.append(runner)
      if runner.below is not None:
        runner.x, runner.y = runner.below.x, runner.below.y - 6
        runner.vx = runner.vy = 0
      else:
        self._move(runner)
      distance = hypot(runner.x - self.barrel_x, runner.y - (93 - self.radius))
      hit = abs(distance - self.radius) < 4.5 + self.radius * 0.05
      escaped = not (0 <= runner.x <= 100 and -50 <= runner.y <= 100)
      if hit or escaped:
        self._detach(runner)
        if runner.above is not None:
          jumped.append(runner.above)
          self._detach(runner.above)
        self.runners.remove(runner)
        if hit:
          self.downed.append(Runner(runner.x, runner.y, runner.vx - 2 * self.barrel_speed, runner.vy))
        else:
          replacements += 1
    self.runners.extend(self._runner() for _ in range(replacements))
    for runner in jumped:
      runner.jumped = True
    for runner in self.downed:
      runner.x += runner.vx
      runner.y += runner.vy
      runner.vy += 0.2
    self.downed = [runner for runner in self.downed if runner.y < 105]
    self.terminal = not self.runners
    self.next_wave = self.barrel_x < -self.radius
    return len(self.runners) if self.next_wave else 0

  @staticmethod
  def _detach(runner):
    if runner.below is not None:
      runner.below.above = None
      runner.below = None

  def _stack(self, runner):
    own = {runner}
    top = runner
    while top.above is not None:
      top = top.above
      own.add(top)
    for other in self.runners:
      if other in own or hypot(runner.x - other.x, runner.y - other.y) >= 4:
        continue
      while other.above is not None:
        other = other.above
      runner.below = other
      other.above = runner
      member = top
      while member is not None:
        member.jumped = member.grounded = False
        member = member.below
      break

  def _move(self, runner):
    runner.x += runner.vx
    runner.y += runner.vy
    runner.vx *= 0.95
    if (runner.x < 7 and runner.vx < 0) or (runner.x >= 77 and runner.vx > 0):
      runner.vx *= -0.5
    runner.vx += (0.01 if runner.x < 50 else -0.01) * sqrt(abs(50 - runner.x) + 1)
    if runner.grounded:
      if runner.x < self.barrel_x:
        runner.vx -= 0.1 * sqrt(self.radius) / sqrt(self.barrel_x - runner.x + 1)
    else:
      runner.vy += 0.1
      if runner.y > 90:
        runner.y, runner.vy = 90, 0
        runner.grounded, runner.jumped = True, False
    if runner.y < 0 and runner.vy < 0:
      runner.vy *= -0.5

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for runner in self.runners:
      if 0 <= runner.x < 100 and 0 <= runner.y < 100:
        channel = 2 if runner.below is not None else int(not runner.grounded)
        state[int(runner.y // 10), int(runner.x // 10), channel] = True
    for runner in self.downed:
      if 0 <= runner.x < 100 and 0 <= runner.y < 100:
        state[int(runner.y // 10), int(runner.x // 10), 4] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 3] = abs(np.hypot(cols - self.barrel_x, rows - (93 - self.radius)) - self.radius) < 7
    for row, value in enumerate((self.steps, self.held, len(self.runners), round(self.radius * 10))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
