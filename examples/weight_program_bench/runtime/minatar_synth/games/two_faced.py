from collections import deque
from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin

import numpy as np


@dataclass
class Piece:
  x: float
  y: float
  angle: float
  side: int = -1


class TwoFacedEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.head = Piece(0, 20, -pi / 2)
    self.item = Piece(0, 0, 0)
    self.turn = 1
    self.speed = self.base_speed = 1.
    self.history = deque(maxlen=256)
    self.tail_count = 3
    self.safe_tail = 2.
    self.multiplier = 3.
    self.action = self.steps = 0
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
    if self.action == 5 and previous != 5:
      self.turn *= -1
      self.speed += 0.1
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.action == 5:
      self.speed += 0.01
    else:
      self.speed += (1 - self.speed) * 0.1
      self.head.angle += self.turn * 0.03 * self.speed * self.base_speed
    self.base_speed *= 1.002
    flips = _move(self.head, 0.5 * self.speed * self.base_speed)
    self.turn *= (-1) ** flips
    self.safe_tail = min(self.tail_count - 1, self.safe_tail + 2 * flips)
    self.history.appendleft(Piece(self.head.x, self.head.y, self.head.angle, self.head.side))
    _move(self.item, 0.2)

    self.safe_tail += (2 - self.safe_tail) * 0.03
    tail = self._tail()
    reward = 0
    if any(_near(segment, self.item, 5) for segment in tail):
      reward = ceil(self.multiplier)
      self.base_speed = 1.
      self.tail_count = min(25, self.tail_count + 1)
      self.multiplier += self.tail_count
      x, y = self.random.uniform(-35, 35, 2)
      self.item = Piece(x, y, self.random.uniform(-pi, pi), -self.head.side)
    self.terminal = any(
      index >= self.safe_tail and _near(self.head, segment, 4)
      for index, segment in enumerate(tail)
    )
    self.multiplier = max(1, self.multiplier - 0.01)
    return reward

  def _tail(self):
    return [self.history[index] for index in range(0, min(len(self.history), 9 * self.tail_count), 9)]

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    _mark(state, self.head, 0)
    _mark(state, self.item, 1 if self.item.side == self.head.side else 2)
    for index, segment in enumerate(self._tail()):
      if segment.side == self.head.side:
        _mark(state, segment, 3 if index >= self.safe_tail else 4)
    values = (self.steps, self.action, self.head.side > 0, self.turn > 0,
              round((self.head.angle % (2 * pi)) * 64),
              round(self.speed * 32), round(self.base_speed * 16),
              self.tail_count, ceil(self.multiplier), round(self.safe_tail * 16))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _move(piece, speed):
  vx, vy = cos(piece.angle), sin(piece.angle)
  piece.x += vx * speed
  piece.y += vy * speed
  flips = 0
  if abs(piece.x) > 40 and piece.x * vx > 0:
    vx *= -1
    flips += 1
  if abs(piece.y) > 40 and piece.y * vy > 0:
    vy *= -1
    flips += 1
  piece.side *= (-1) ** flips
  piece.angle = atan2(vy, vx)
  return flips


def _near(first, second, radius):
  return first.side == second.side and hypot(first.x - second.x, first.y - second.y) <= radius


def _cell(piece):
  return tuple(int(np.clip((value + 40) // 8, 0, 9)) for value in (piece.y, piece.x))


def _mark(state, piece, channel):
  row, col = _cell(piece)
  state[row, col, channel] = True
