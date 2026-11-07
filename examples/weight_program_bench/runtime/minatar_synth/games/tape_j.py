from dataclasses import dataclass
from math import floor, hypot, sqrt

import numpy as np


@dataclass
class Block:
  x: float
  width: float
  height: float


@dataclass
class Fire:
  x: float
  y: float
  vx: float
  vy: float
  size: float


class TapeJEnvironment:
  score_cap = 4000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.tip = np.array([100., 89.])
    self.base = self.tip.copy()
    self.mode = "ground"
    self.block = None
    self.scroll = self.distance = self.block_timer = 0.
    self.fire_timer = 100.
    self.steps = 0
    self.held = self.terminal = False
    self.blocks, self.tapes, self.fires = [], [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _corner(self, mode):
    self.tapes.append((self.base.copy(), self.tip.copy()))
    self.base = self.tip.copy()
    self.mode = mode

  def _pull(self):
    self.distance += 1
    if self.mode in ("ground", "top"):
      self.tip[0] += 1
    if self.mode == "up":
      self.tip[1] -= 1
      height = 90 - self.block.height
      if self.tip[1] < height:
        self.tip[1] = height
        self._corner("top")
    elif self.mode == "top" and self.tip[0] > self.block.x + self.block.width + 1:
      self.tip[0] = self.block.x + self.block.width + 1
      self._corner("down")
    elif self.mode == "down":
      self.tip[1] += 1
      if self.tip[1] > 89:
        self.tip[1] = 89
        self._corner("ground")
        self.tip[0] += 3
        self.block = None

  def _advance(self):
    reward = 0
    if self.held:
      self._pull()
    else:
      if self.distance:
        reward = floor(sqrt(self.distance) * self.distance * 0.1 + 1)
        self.distance = 0
      start, end = self.tapes[0] if self.tapes else (self.base, self.tip)
      start += (end - start) * 0.2
      if self.tapes and np.linalg.norm(end - start) < 3:
        self.tapes.pop(0)
    self.terminal = self.tip[0] < 0
    for start, end in [*self.tapes, (self.base, self.tip)]:
      start[0] = max(0, start[0] - self.scroll)
      end[0] -= self.scroll
    self.tapes = [(start, end) for start, end in self.tapes if end[0] >= 0]
    self.scroll = 0.1 + max(0, self.tip[0] - 50) * 0.1
    self.block_timer -= self.scroll
    if self.block_timer < 0:
      width = self.random.uniform(20, 40)
      self.blocks.append(Block(200, width, self.random.uniform(10, 50)))
      self.block_timer = width + self.random.uniform(20, 70)
    for block in self.blocks:
      if self.mode == "ground" and block.x - 1.5 < self.tip[0] < block.x + block.width + 1.5:
        self.tip[:] = block.x - 1, 89
        self._corner("up")
        self.block = block
      block.x -= self.scroll
    self.blocks = [block for block in self.blocks if block.x + block.width >= 0]
    self.fire_timer -= 1
    if self.fire_timer < 0:
      size = self.random.uniform(5, 15)
      self.fires.append(Fire(self.random.uniform(150, 220), -size,
                             -self.random.uniform(0.5, 1), self.random.uniform(0.8, 1.2), size))
      self.fire_timer = self.random.uniform(40, 60)
    points = self._tape_points()
    for fire in self.fires:
      fire.x += fire.vx - self.scroll
      fire.y += fire.vy
      self.terminal |= bool(np.any((abs(points[:, 0] - fire.x) < fire.size / 2 + 1.5)
                                   & (abs(points[:, 1] - fire.y) < fire.size / 2 + 1.5)))
    self.fires = [fire for fire in self.fires if fire.y <= 90]
    return reward

  def _tape_points(self):
    segments = [*self.tapes, (self.base, self.tip)]
    return np.concatenate([np.linspace(start, end, max(2, int(hypot(*(end - start))) + 1))
                           for start, end in segments])

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, points in ((0, self.tip[None]), (1, self._tape_points()),
                            (4, (self.tapes[0][0] if self.tapes else self.base)[None])):
      visible = points[(points[:, 0] >= 0) & (points[:, 0] < 200)]
      cols, rows = (visible // (20, 10)).astype(int).T
      state[rows, cols, channel] = True
    rows, cols = np.ogrid[:100:10, :200:20]
    state[9, :, 2] = True
    for block in self.blocks:
      state[:, :, 2] |= (cols < block.x + block.width) & (cols + 20 > block.x) & (rows + 10 > 90 - block.height)
    for fire in self.fires:
      state[:, :, 3] |= (abs(cols + 10 - fire.x) < fire.size / 2 + 10) & (
        abs(rows + 5 - fire.y) < fire.size / 2 + 5)
    values = (self.steps, int(self.distance), self.held, ("ground", "up", "top", "down").index(self.mode))
    for row, value in enumerate(values):
      for bit in range(11):
        state[row + bit // 10 * 4, bit % 10, 5] = bool(value & (1 << bit))
    return state
