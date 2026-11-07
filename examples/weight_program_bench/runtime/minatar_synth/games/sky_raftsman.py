from dataclasses import dataclass
from math import atan2, cos, floor, hypot, sin

import numpy as np


@dataclass
class Log:
  x: float
  y: float
  size: float
  spin: float
  angle: float = 0
  vx: float = 0
  vy: float = 0
  distance: float = 0


class SkyRaftsmanEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = self.vx = self.vy = self.distance = 0.
    self.steps = 0
    self.jumping = self.held = self.terminal = False
    self.current = Log(50, 75, 20, -0.05)
    self.logs = [self.current, *(self._log(75 + i * 30) for i in range(1, 5))]

  def _log(self, y):
    return Log(self.random.uniform(10, 90), y, self.random.uniform(15, 25),
               self.random.uniform(0.04, 0.08) * self.random.choice((-1, 1)))

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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
    scroll = -0.05 + 0.1 * max(0, 20 - self.y) - 0.2 * max(0, self.y - 50)
    log = self.current
    if tap and not self.jumping:
      self.jumping = True
      self.vx, self.vy = 2 * cos(log.angle), 2 * sin(log.angle)
      self.x += 2 * self.vx
      self.y += 2 * self.vy
      log.vx, log.vy = -self.vx, -self.vy
    if self.jumping:
      self.x += self.vx
      self.y += self.vy + scroll
      self.vx *= 0.99
      self.vy = self.vy * 0.99 + (0.05 if self.held else 0.1)
    else:
      self.x = log.x + log.size / 2 * cos(log.angle)
      self.y = log.y + log.size / 2 * sin(log.angle)
    if (self.x < 3 and self.vx < 0) or (self.x > 97 and self.vx > 0):
      self.vx *= -1
    self.terminal = self.y > 100
    reward = 0
    for log in self.logs:
      log.x += log.vx
      log.y += log.vy + scroll
      log.distance += log.vy
      if log.distance > 20:
        log.distance -= 20
        reward += max(0, floor(log.vy * 5))
      log.vx *= 0.99
      log.vy *= 0.99
      if (log.x < log.size / 2 and log.vx < 0) or (log.x > 100 - log.size / 2 and log.vx > 0):
        log.vx *= -1
      log.angle += log.spin
    self.logs = [log for log in self.logs if log.y >= -99 or log is self.current]
    self.distance += scroll
    if self.distance < 0 and not self.jumping:
      self.logs.append(self._log(125))
      self.distance = 100
    if self.jumping:
      for log in self.logs:
        if log.y <= 100 + log.size / 2 and abs(
          hypot(self.x - log.x, self.y - log.y) - log.size / 2
        ) < 5:
          self.jumping = False
          self.current = log
          log.angle = atan2(self.y - log.y, self.x - log.x)
          break
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = np.clip(np.array([self.y, self.x]) // 10, 0, 9).astype(int)
    state[row, col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for log in self.logs:
      if log.y > 100 + log.size / 2:
        state[9, :, 3] |= abs(cols[0] - log.x) < log.size / 2 + 5
      else:
        channel = 2 if log is self.current and not self.jumping else 1
        state[:, :, channel] |= abs(np.hypot(cols - log.x, rows - log.y) - log.size / 2) < 6
    dx, dy = (self.vx, self.vy) if self.jumping else (cos(self.current.angle), sin(self.current.angle))
    row, col = np.clip(np.array([self.y + 10 * dy, self.x + 10 * dx]) // 10, 0, 9).astype(int)
    state[row, col, 4] = True
    values = (self.steps, self.held, self.jumping,
              round((self.vx + 10) * 20), round((self.vy + 10) * 20))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
