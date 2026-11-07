from dataclasses import dataclass
from math import cos

import numpy as np


@dataclass
class Motion:
  pos: np.ndarray
  vel: np.ndarray


@dataclass
class Enemy:
  motion: Motion
  sink: np.ndarray
  target: Motion


class SightOnEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.sight = Motion(np.full(2, 50.), np.zeros(2))
    self.enemies, self.targets, self.blasts = [], [], []
    self.steps = self.count = self.wave_timer = self.enemy_timer = 0
    self.multiplier = 1
    self.penalty = 0.
    self.held = self.terminal = False

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

  def _wave(self):
    target = Motion(np.array([self.random.uniform(20, 80), 0.]),
                    np.array([self.random.uniform(-0.1, 0.1), 0.1]))
    self.targets.append(target)
    self.template = Enemy(
      Motion(np.array([self.random.uniform(10, 90), -5.]),
             self.random.uniform(1, 2, 2) * self.random.choice((-1, 1), 2)),
      self.random.uniform(0.02, 0.07, 2), target,
    )
    self.count = self.random.integers(2, 5)
    self.interval = self.random.uniform(3, 9)
    self.enemy_timer = 0
    self.wave_timer = self.random.uniform(120, 150)

  def _advance(self, tap):
    self.wave_timer -= 1
    if self.wave_timer < 0:
      self._wave()
    for target in self.targets:
      target.pos += target.vel
      target.pos[0] %= 99
    self.targets = [target for target in self.targets if target.pos[1] <= 150]
    template = self.template
    template.motion.pos += template.target.vel
    self.enemy_timer -= 1
    if self.count and self.enemy_timer < 0:
      self.enemies.append(Enemy(Motion(template.motion.pos.copy(), template.motion.vel.copy()),
                                template.sink, template.target))
      self.count -= 1
      self.enemy_timer = self.interval
    if not self.enemies and not self.count:
      self.wave_timer = 0
    reward = 0
    remaining = []
    for enemy in self.enemies:
      motion = enemy.motion
      delta = enemy.target.pos - motion.pos
      delta[0] = (delta[0] + 50) % 100 - 50
      motion.vel = (motion.vel + delta * enemy.sink) * 0.997
      motion.pos += motion.vel * 0.2
      motion.pos[0] %= 99
      if any(np.all(abs(motion.pos - center) < 3 + 7.5 * cos(age * 0.05))
             for center, age in self.blasts):
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= bool(motion.pos[1] > 90)
        remaining.append(enemy)
    self.enemies = remaining
    self.blasts = [(center, age + 1) for center, age in self.blasts if age < 31]
    leader = max(self.enemies, key=lambda enemy: enemy.motion.pos[1], default=None)
    target = leader.motion.pos if leader is not None and leader.motion.pos[1] > 0 else np.full(2, 50.)
    delta = target - self.sight.pos
    delta[0] = (delta[0] + 50) % 100 - 50
    self.sight.vel = (self.sight.vel + delta * 0.01) * 0.97
    self.sight.pos += self.sight.vel * 0.2
    self.sight.pos[0] %= 99
    if tap:
      self.multiplier = 1
      self.blasts.append((self.sight.pos.copy(), 0))
      for target in self.targets:
        target.pos[1] += self.penalty
      self.penalty += 1
      self.sight.vel *= 0.1
    self.penalty *= 0.99
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    positions = [(self.sight.pos, 0)]
    positions.extend((enemy.motion.pos, 1) for enemy in self.enemies)
    positions.extend((target.pos, 2) for target in self.targets)
    for (x, y), channel in positions:
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), channel] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for (x, y), age in self.blasts:
      reach = 7.5 * cos(age * 0.05) + 5
      state[:, :, 3] |= (abs(cols - x) < reach) & (abs(rows - y) < reach)
    state[9, :, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.held, round(self.penalty * 10))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
