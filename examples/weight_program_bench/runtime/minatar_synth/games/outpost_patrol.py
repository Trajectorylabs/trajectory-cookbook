from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Wave:
  angle: float
  width: float = 0


def position(angle):
  return 50 + 40 * math.cos(angle), 50 + 40 * math.sin(angle)


def separation(a, b):
  return abs((a - b + math.pi) % math.tau - math.pi)


class OutpostPatrolEnvironment:
  score_cap = 12

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = self.obstacle_timer = self.bomb_timer = 0.0
    self.jump = self.steps = 0
    self.height = 0.0
    self.obstacles = []
    self.bombs = []
    self.waves: list[Wave] = []
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    self.angle = (self.angle + 0.02) % math.tau
    if self.jump or tap:
      self.jump += 1 if self.pressed else 2
      if self.jump >= 30:
        self.jump = 0
    self.height = 10 * math.sin(self.jump * math.pi / 30)
    self.obstacle_timer -= 1
    if self.obstacle_timer < 0:
      opposite = self.angle + math.pi
      if all(separation(o, opposite) > math.pi * 0.2 for o in self.obstacles):
        self.obstacles.append(opposite)
      self.obstacle_timer += self.random.uniform(60, 150)
    self.obstacles = [o - 0.01 for o in self.obstacles]
    self.bomb_timer -= 1
    if self.bomb_timer < 0:
      self.bombs.append(self.random.uniform(0, math.tau))
      self.bomb_timer += self.random.uniform(90, 120)
    self.bombs = [b - 0.015 for b in self.bombs]
    for wave in self.waves:
      wave.width += 0.1
    survivors = [o for o in self.obstacles if all(
      separation(o, wave.angle) > wave.width / 2 + 0.09 for wave in self.waves
    )]
    reward = len(self.obstacles) - len(survivors)
    self.obstacles = survivors
    self.waves = [wave for wave in self.waves if wave.width <= 1]
    x, y = position(self.angle)
    if not self.jump:
      for obstacle in self.obstacles:
        ox, oy = position(obstacle)
        self.terminal |= abs(x - ox) < 6 and abs(y - oy) < 6
    else:
      remaining = []
      for bomb in self.bombs:
        bx, by = position(bomb)
        if abs(x - bx) < 6.5 and abs(y - self.height - by + 10) < 6.5:
          self.waves.append(Wave(bomb))
        else:
          remaining.append(bomb)
      self.bombs = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = position(self.angle)
    state[int((y - self.height) // 10), int(x // 10), 0] = True
    for channel, angles in ((1, self.obstacles), (2, self.bombs)):
      for angle in angles:
        x, y = position(angle)
        state[int((y - (10 if channel == 2 else 0)) // 10), int(x // 10), channel] = True
    for wave in self.waves:
      for angle in np.linspace(wave.angle - wave.width / 2, wave.angle + wave.width / 2, 10):
        x, y = position(angle)
        state[int(y // 10), int(x // 10), 3] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 4] = abs(np.hypot(cols - 50, rows - 50) - 40) < 5
    for row, value in enumerate((self.steps, self.jump, round(self.angle * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
