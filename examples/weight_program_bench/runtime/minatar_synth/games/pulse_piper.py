from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
  x: float
  width: float
  height: float
  gap: float
  dot: bool


class PulsePiperEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 10.0
    self.charge = self.steps = self.spawn_timer = 0
    self.held = self.released = self.terminal = False
    self.obstacles = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance(held)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, held):
    self.x += 0.7
    if self.x > 100:
      self.x = 0
    self.released = self.held and not held
    discharge = self.charge if self.released else 0
    self.held = held
    self.charge = min(99, self.charge + 1) if held else 0
    offset = 2 + self.charge / 2
    reward = 0
    remaining = []
    for obstacle in self.obstacles:
      obstacle.x -= 0.5
      overlap = (
        abs(obstacle.x - self.x) < 4 + obstacle.width / 2
        and offset + 2 > obstacle.gap
        and offset - 2 < obstacle.gap + obstacle.height
      )
      if overlap and obstacle.dot and self.released:
        reward += discharge
        continue
      if overlap and not obstacle.dot and not self.released:
        self.terminal = True
      if obstacle.x + obstacle.width / 3 >= 0:
        remaining.append(obstacle)
    self.obstacles = remaining
    if self.spawn_timer == 0:
      self._spawn()
      self.spawn_timer = 60
    self.spawn_timer -= 1
    return reward

  def _spawn(self):
    dot = self.random.random() < 0.7
    wall = self.random.random() < 0.3
    if dot:
      self.obstacles.append(Obstacle(102, 4, 2, 0, True))
    else:
      close = not wall and self.x < 40 and self.random.random() < 0.65
      gap = 0 if close else self.random.uniform(10, 20)
      self.obstacles.append(Obstacle(104.5, 9, 50 if wall else 4, gap, False))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[:100:10, :100:10]
    for obstacle in self.obstacles:
      horizontal = (cols < obstacle.x + obstacle.width / 2) & (cols + 10 > obstacle.x - obstacle.width / 2)
      top = (rows < 50 - obstacle.gap) & (rows + 10 > 50 - obstacle.gap - obstacle.height)
      bottom = (rows < 50 + obstacle.gap + obstacle.height) & (rows + 10 > 50 + obstacle.gap)
      channel = 2 if obstacle.dot else 3 if obstacle.height < 50 else 4
      state[:, :, channel] |= horizontal & (top | bottom)
    offset = 2 + self.charge / 2
    for y in (50 - offset, 50 + offset):
      state[int(np.clip(y // 10, 0, 9)), min(9, int(self.x // 10)), int(self.released)] = True
    for row, value in enumerate((self.charge, self.spawn_timer, self.steps, self.held)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
