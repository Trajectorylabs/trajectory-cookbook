from dataclasses import dataclass

import numpy as np


SIZES = (5, 8, 12)


@dataclass
class Block:
  x: float
  y: float
  vx: float
  vy: float
  size: int
  squash: float = 1.
  angle: float = 0.


class SplitzigEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 80.
    self.direction = 1
    self.squash = 1.
    self.tilt = self.stack = 0.
    self.frame = self.steps = self.timer = 0
    self.held = self.terminal = False
    self.blocks, self.bullets = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.direction *= -1
      self.squash = 0.5
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.squash += (1 - self.squash) * 0.15
    self.tilt += (self.direction * 0.3 - self.tilt) * 0.2
    self.x = (self.x + self.direction * 1.5) % 100
    self.stack *= 0.998
    self.y = 90 - self.stack
    self.terminal = self.y < 0
    if self.frame % 7 == 0:
      self.bullets.append((self.x, self.y - 5))
    self.bullets = [(x, y - 3) for x, y in self.bullets if y - 3 >= -5]
    self.timer += 1
    if self.timer >= self.random.uniform(88, 111):
      self.timer = 0
      self.blocks.append(Block(self.random.uniform(15, 85), -10,
                               self.random.uniform(-0.3, 0.3), self.random.uniform(0.2, 0.8), 2))
    reward = 0
    spent = set()
    remaining = []
    for block in self.blocks:
      block.x += block.vx
      block.y += block.vy
      if (block.x < 5 and block.vx < 0) or (block.x > 95 and block.vx > 0):
        block.vx *= -1
        block.squash = 0.6
      block.squash += (1 - block.squash) * 0.1
      block.angle += block.vx * 0.05
      size = SIZES[block.size]
      c, s = abs(np.cos(block.angle)), abs(np.sin(block.angle))
      half_x = size * (c / block.squash + s * block.squash) / 2
      half_y = size * (s / block.squash + c * block.squash) / 2
      hits = [index for index, (x, y) in enumerate(self.bullets)
              if abs(block.x - x) < half_x + 1 and abs(block.y - y) < half_y + 2]
      if hits:
        spent.update(hits)
        reward += 3 - block.size
        if block.size:
          for direction in (-1, 1):
            remaining.append(Block(block.x + 5 * direction, block.y,
                                   direction * self.random.uniform(0.8, 1.1),
                                   block.vy * 0.8, block.size - 1, 0.6, block.angle))
      elif abs(block.x - self.x) < half_x + 3 / self.squash and abs(block.y - self.y) < half_y + 3:
        self.terminal = True
      elif block.y > 100 - self.stack - size / 2:
        self.stack += size / 2
      else:
        remaining.append(block)
    self.blocks = remaining
    self.bullets = [bullet for index, bullet in enumerate(self.bullets) if index not in spent]
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(self.x // 10), 0] = True
    for x, y in self.bullets:
      if 0 <= y < 100:
        state[int(y // 10), int(x // 10), 1] = True
    for block in self.blocks:
      if 0 <= block.x < 100 and 0 <= block.y < 100:
        state[int(block.y // 10), int(block.x // 10), block.size + 2] = True
    rows = np.arange(10) * 10
    state[rows + 10 > 100 - self.stack, :, 5] = True
    values = (self.steps, self.direction > 0, self.held, self.frame % 7, self.timer)
    state[:5, :9, 5] = False
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
