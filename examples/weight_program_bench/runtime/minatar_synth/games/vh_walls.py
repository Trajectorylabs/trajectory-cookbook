from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass
class Wall:
  x: float
  y: float
  horizontal: bool
  fixed: bool = False

  def bounds(self):
    length = 50 if self.fixed else 10
    return (length, 1) if self.horizontal else (1, length)


class VhWallsEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.walls = [Wall(1, 50, False, True), Wall(99, 50, False, True),
                  Wall(50, 1, True, True), Wall(50, 99, True, True)]
    self.x = self.y = self.cursor_x = self.cursor_y = 50.
    self.vx = self.vy = 1
    self.horizontal = False
    self.target_x = self.target_y = 50.
    self._new_target()
    self.steps = self.action = 0
    self.terminal = False

  def _new_target(self):
    previous = self.target_x, self.target_y
    for _ in range(9):
      self.target_x, self.target_y = self.random.uniform(20, 80, 2)
      if hypot(self.target_x - previous[0], self.target_y - previous[1]) > 40:
        break
    self.radius = 20.
    self.warmup = 16

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    dx, dy = {1: (-5, 0), 2: (0, -5), 3: (5, 0), 4: (0, 5)}.get(self.action, (0, 0))
    self.cursor_x = min(98, max(1, self.cursor_x + dx))
    self.cursor_y = min(98, max(1, self.cursor_y + dy))
    place = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(place and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, place):
    self.x = min(99, max(1, self.x + self.vx))
    self.y = min(99, max(1, self.y + self.vy))
    if place:
      self.walls.insert(0, Wall(self.cursor_x, self.cursor_y, self.horizontal))
      self.horizontal = not self.horizontal
    warming = self.warmup > 0
    target = self.target_x, self.target_y, self.radius
    reward = 0
    if warming:
      self.warmup -= 1
      self.radius += (5 - self.radius) * 0.2
    elif _hits_ring(target, self.x, self.y, 2.5, 2.5):
      reward = len(self.walls) - 4
      self._new_target()
      self.radius += 0.02
    else:
      self.radius += 0.02

    remaining = []
    ball_x, ball_y = self.x, self.y
    for wall in self.walls:
      width, height = wall.bounds()
      hits_target = _hits_ring(target, wall.x, wall.y, width, height)
      if not wall.fixed and ((warming and hits_target) or any(
        _walls_overlap(wall, other) for other in remaining if not other.fixed
      )):
        continue
      if not warming and hits_target:
        self.terminal = True
      if abs(wall.x - ball_x) < width + 2.5 and abs(wall.y - ball_y) < height + 2.5:
        if wall.horizontal:
          self.vy *= -1
          self.y = wall.y + self.vy * 4
        else:
          self.vx *= -1
          self.x = wall.x + self.vx * 4
      remaining.append(wall)
    self.walls = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    _mark(state, self.x, self.y, 0)
    for wall in self.walls:
      width, height = wall.bounds()
      state[:, :, 2 if wall.horizontal else 1] |= (
        (abs(cols - wall.x) < width + 5) & (abs(rows - wall.y) < height + 5)
      )
    ring = abs(np.hypot(cols - self.target_x, rows - self.target_y) - self.radius) <= 6
    state[:, :, 3] = ring
    _mark(state, self.cursor_x, self.cursor_y, 4)
    values = (self.steps, self.action, self.horizontal, self.vx > 0, self.vy > 0,
              self.warmup, round(self.radius * 16), len(self.walls) - 4)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _hits_ring(target, x, y, width, height):
  tx, ty, radius = target
  dx, dy = abs(x - tx), abs(y - ty)
  near = hypot(max(0, dx - width), max(0, dy - height))
  far = hypot(dx + width, dy + height)
  return near <= radius + 1 and far >= radius - 1


def _walls_overlap(first, second):
  aw, ah = first.bounds()
  bw, bh = second.bounds()
  return abs(first.x - second.x) < aw + bw and abs(first.y - second.y) < ah + bh


def _mark(state, x, y, channel):
  state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), channel] = True
