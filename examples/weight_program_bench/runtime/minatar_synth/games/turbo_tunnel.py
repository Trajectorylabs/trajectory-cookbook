from dataclasses import dataclass
from math import cos, hypot, pi, sin

import numpy as np


@dataclass
class Obstacle:
  angle: float
  radius: float
  speed: float
  fuse: int = -1

  def position(self):
    return _position(self.angle, self.radius)


class TurboTunnelEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle, self.radius, self.direction = pi / 2, 30., 1
    self.obstacles = []
    self.multiplier = 1
    self.steps = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    turn = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(turn and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, turn):
    spawn = turn
    if turn:
      self.direction *= -1
    self.angle = (self.angle + 0.5 / self.radius) % (2 * pi)
    self.radius += self.direction * 0.4
    if (self.radius > 50 and self.direction > 0) or (self.radius < 10 and self.direction < 0):
      self.direction *= -1
      spawn = True

    player = _position(self.angle, self.radius)
    armed = []
    loose = []
    remaining = []
    reward = 0
    had_fuse = False
    for obstacle in self.obstacles:
      obstacle.angle += obstacle.speed / (obstacle.radius * 0.03)
      point = obstacle.position()
      if obstacle.fuse > 0:
        had_fuse = True
        obstacle.fuse -= 1
        if obstacle.fuse == 0:
          reward += self.multiplier
          self.multiplier += 1
          continue
        armed.append(point)
      elif any(_distance(point, other) <= 3.5 for other in loose):
        obstacle.fuse = 30
        armed.append(point)
      else:
        loose.append(point)
        if _distance(point, player) <= 2.5:
          self.terminal = True
      remaining.append(obstacle)
    self.obstacles = remaining
    for obstacle in self.obstacles:
      if obstacle.fuse < 0 and any(_distance(obstacle.position(), point) <= 4.5 for point in armed):
        obstacle.fuse = 30
        armed.append(obstacle.position())
        had_fuse = True
    if not had_fuse:
      self.multiplier = 1

    if spawn and not self.terminal:
      obstacle = Obstacle(self.angle - 3 / self.radius, self.radius, -self.random.uniform(0.01, 0.03))
      x, y = obstacle.position()
      if not any(abs(x - ox) < 6.5 and abs(y - oy) < 6.5
                 for ox, oy in (other.position() for other in self.obstacles)):
        self.obstacles.append(obstacle)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    _mark(state, _position(self.angle, self.radius), 0)
    for obstacle in self.obstacles:
      _mark(state, obstacle.position(), 2 if obstacle.fuse > 0 else 1)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    radius = np.hypot(cols - 50, rows - 50)
    state[:, :, 3] = (abs(radius - 10) < 6) | (abs(radius - 50) < 6)
    _mark(state, _position(self.angle + 2 / self.radius, self.radius + self.direction * 1.6), 4)
    values = (self.steps, self.action, self.direction > 0, self.multiplier,
              round(self.radius * 8), round(self.angle * 64))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _position(angle, radius):
  return 50 + cos(angle) * radius, 50 + sin(angle) * radius


def _distance(first, second):
  return hypot(first[0] - second[0], first[1] - second[1])


def _mark(state, point, channel):
  x, y = point
  state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), channel] = True
