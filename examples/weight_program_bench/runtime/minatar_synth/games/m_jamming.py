from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Missile:
  x: float
  y: float
  angle: float
  speed: float = 0.4
  payload: float = -1


@dataclass
class Blast:
  x: float
  y: float
  target: float
  radius: float = 1
  growth: float = 0.5


class MJammingEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 50.0
    self.vx = self.vy = self.radius = self.timer = 0.0
    self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.missiles: list[Missile] = []
    self.blasts: list[Blast] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    release = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, release):
    sx, sy = (50 - self.x) * 0.05, (50 - self.y) * 0.05
    for blast in self.blasts:
      blast.radius += blast.growth
      if blast.radius > blast.target:
        blast.growth = -0.5
    self.blasts = [b for b in self.blasts if b.radius >= 1]
    self.vx *= 0.98
    self.vy *= 0.98
    self.x += self.vx + sx
    self.y += self.vy + sy
    if self.pressed or release:
      self.radius *= 1.01
      self.vx *= 0.9
      self.vy *= 0.9
    else:
      self.radius = min(15, self.radius + 0.5)
    reward = self._jam() if release or self.radius > 50 else 0
    self.timer -= 1
    if self.timer < 0:
      self._spawn()
    survivors = []
    for missile in self.missiles:
      if missile.payload < 0:
        target = math.atan2(self.y - missile.y, self.x - missile.x)
        delta = (target - missile.angle + math.pi) % math.tau - math.pi
        missile.angle += min(0.005, max(-0.005, delta))
      missile.x += math.cos(missile.angle) * missile.speed + sx
      missile.y += math.sin(missile.angle) * missile.speed + sy
      if any(abs(math.hypot(missile.x - b.x, missile.y - b.y) - b.radius) < 4 for b in self.blasts):
        if missile.payload > 1:
          self.blasts.append(Blast(missile.x, missile.y, missile.payload))
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= math.hypot(missile.x - self.x, missile.y - self.y) < 4
        if -100 <= missile.x < 100 and -100 <= missile.y < 100:
          survivors.append(missile)
    hostile = [m for m in survivors if m.payload < 0]
    remaining = []
    for missile in survivors:
      if missile.payload > 0 and any(math.hypot(missile.x - m.x, missile.y - m.y) < 3 for m in hostile):
        self.blasts.append(Blast(missile.x, missile.y, missile.payload))
        reward += self.multiplier
        self.multiplier += 1
      else:
        remaining.append(missile)
    self.missiles = remaining
    nearest = min(survivors, key=lambda m: math.hypot(m.x - self.x, m.y - self.y), default=None)
    if nearest is not None and math.hypot(nearest.x - self.x, nearest.y - self.y) < 50:
      angle = math.atan2(self.y - nearest.y, self.x - nearest.x) + math.pi / 3
      self.vx += math.cos(angle) * 0.01
      self.vy += math.sin(angle) * 0.01
    return reward

  def _jam(self):
    self.multiplier = 1
    targets = [m for m in self.missiles if m.payload < 0
               and math.hypot(m.x - self.x, m.y - self.y) < self.radius + 1]
    count = len(targets)
    for missile in targets:
      missile.angle = math.atan2(missile.y - self.y, missile.x - self.x)
      missile.payload = min(25, count ** 1.5)
      missile.speed *= 0.5
    self.radius = count
    self.multiplier += count
    return count * (count + 1) // 2

  def _spawn(self):
    for _ in range(int(self.random.integers(6, 9))):
      angle = self.random.uniform(0, math.tau)
      x, y = 50 + math.cos(angle) * 99, 50 + math.sin(angle) * 99
      tx, ty = self.random.uniform(40, 60, 2)
      self.missiles.append(Missile(x, y, math.atan2(ty - y, tx - x)))
    self.timer = self.random.uniform(60, 70)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), min(9, max(0, int(self.x // 10))), 0] = True
    for missile in self.missiles:
      if 0 <= missile.x < 100 and 0 <= missile.y < 100:
        state[int(missile.y // 10), int(missile.x // 10), 1 + (missile.payload > 0)] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for blast in self.blasts:
      state[:, :, 3] |= abs(np.hypot(cols - blast.x, rows - blast.y) - blast.radius) < 6
    state[:, :, 4] = np.hypot(cols - self.x, rows - self.y) < self.radius + 1
    for row, value in enumerate((self.steps, min(511, self.multiplier), round(self.radius * 8))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
