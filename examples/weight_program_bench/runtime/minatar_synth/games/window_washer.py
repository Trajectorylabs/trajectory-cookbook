from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
  x: float
  y: float
  clean: bool = False


@dataclass
class Obstacle:
  x: float
  y: float
  vx: float


class WindowWasherEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vx = 50., 10., 1
    self.windows = [Window(self.random.uniform(15, 85), 20 + i * 30) for i in range(10)]
    self.obstacles = []
    self.distance = 0.
    self.scroll = 0.5
    self.multiplier = 1
    self.steps = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.x += self.vx
    if self.x < 10 or self.x > 90:
      self.vx *= -1
    self.y = min(95, max(5, self.y + (-1.5 if self.action == 5 else 1)))
    reward = 0
    for window in self.windows:
      window.y -= self.scroll
      if window.y < -30:
        window.y += 330
        if not window.clean:
          self.multiplier = max(1, self.multiplier - 1)
        window.clean = False
      if not window.clean and self._overlaps(window.x, window.y, 7.5, 10):
        window.clean = True
        reward += self.multiplier
        self.multiplier += 1

    for obstacle in self.obstacles:
      obstacle.y -= self.scroll
      obstacle.x += obstacle.vx
      if obstacle.x < 5 or obstacle.x > 95:
        obstacle.vx *= -1
      half_width, half_height = (3, 3) if obstacle.vx else (5, 7)
      self.terminal |= self._overlaps(obstacle.x, obstacle.y, half_width, half_height)
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y >= -10]
    self.distance -= self.scroll
    if self.distance < 0:
      self._spawn()
    self.scroll = 0.6
    return reward

  def _overlaps(self, x, y, half_width, half_height):
    return abs(x - self.x) < 10 + half_width and abs(y - self.y) < 2.5 + half_height

  def _spawn(self):
    bird = self.random.random() < 0.5
    x = self.random.uniform(15, 85)
    bird |= any(not window.clean and abs(window.x - x) < 14.5 and abs(window.y - 110) < 19
                for window in self.windows)
    speed = self.random.choice((-0.5, 0.5)) if bird else 0.
    self.obstacles.append(Obstacle(x, 110, speed))
    self.distance += self.random.uniform(40, 50)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 0] = (abs(rows - self.y) < 7.5) & (abs(cols - self.x) < 15)
    for window in self.windows:
      state[:, :, 2 if window.clean else 1] |= (
        (abs(rows - window.y) < 15) & (abs(cols - window.x) < 12.5)
      )
    for obstacle in self.obstacles:
      half_width, half_height = (3, 3) if obstacle.vx else (5, 7)
      state[:, :, 3 if obstacle.vx else 4] |= (
        (abs(rows - obstacle.y) < half_height + 5)
        & (abs(cols - obstacle.x) < half_width + 5)
      )
    values = (self.steps, self.action, self.vx > 0, self.multiplier, round(self.distance * 4),
              round(self.x * 4), round(self.y * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
