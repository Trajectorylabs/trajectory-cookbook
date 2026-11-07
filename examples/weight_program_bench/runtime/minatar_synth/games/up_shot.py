from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Spawner:
  x: float
  size: float
  acceleration: float
  interval: float
  variation: float
  timer: float


@dataclass
class Rock:
  x: float
  y: float
  size: float
  acceleration: float
  vy: float = 0.


@dataclass
class Shot:
  x: float
  y: float = 87.


class UpShotEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.vx = 10., 0.
    self.spawners, self.rocks, self.shots = [], [], []
    self.spawn_distance = 0.
    self.cooldown = self.steps = self.action = 0
    self.multiplier = 1
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
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(previous if frame == 0 else self.action)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, previous):
    scroll = 0.1 + max(0, self.x - 30) * 0.05
    self.cooldown -= 1
    if self.action == 5:
      if previous != 5:
        self.vx = 0
        self.multiplier = 1
      if self.cooldown < 0:
        self.shots.append(Shot(self.x))
        self.cooldown = 10
    elif previous == 5:
      self.vx = 1.2
    self.x += self.vx - scroll
    for shot in self.shots:
      shot.x -= scroll
      shot.y -= 2
    self.shots = [shot for shot in self.shots if shot.y >= 0]
    self._spawn_rocks(scroll)

    reward = 0
    remaining = []
    for rock in self.rocks:
      rock.vy += rock.acceleration
      rock.x -= scroll
      rock.y += rock.vy
      if any(_hit(shot, rock) for shot in self.shots):
        rock.size *= 0.7
        if rock.size < 5:
          reward += 10 * self.multiplier
          continue
        reward += self.multiplier
        self.multiplier += 1
      if rock.y <= 90 - rock.size / 2:
        remaining.append(rock)
    self.rocks = remaining
    self.terminal = self.x < -2 or any(
      abs(rock.x - self.x) < rock.size / 2 + 3 and abs(rock.y - 87) < rock.size / 2 + 3
      for rock in self.rocks
    )
    self.shots = [shot for shot in self.shots if not any(_hit(shot, rock) for rock in self.rocks)]
    return reward

  def _spawn_rocks(self, scroll):
    self.spawn_distance -= scroll
    if self.spawn_distance < 0:
      size = self.random.uniform(5, 15)
      speed = self.random.uniform(5, 10) / sqrt(size)
      interval = self.random.uniform(10, 50) * sqrt(size / speed)
      self.spawners.append(Spawner(200, size, speed * 0.01, interval,
                                   self.random.uniform(0.3, 0.9), self.random.uniform(0, interval)))
      self.spawn_distance += self.random.uniform(50, 60)
    for spawner in self.spawners:
      spawner.x -= scroll
      spawner.timer -= 1
      if spawner.timer < 0:
        self.rocks.append(Rock(spawner.x, -spawner.size / 2, spawner.size, spawner.acceleration))
        spawner.timer = spawner.interval * (1 + self.random.uniform(-spawner.variation, spawner.variation))
    self.spawners = [spawner for spawner in self.spawners if spawner.x >= 0]

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, int(np.clip(self.x // 10, 0, 9)), 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for rock in self.rocks:
      mask = (abs(cols - rock.x) < rock.size / 2 + 5) & (abs(rows - rock.y) < rock.size / 2 + 5)
      state[:, :, 1 if rock.size < 8 else 2] |= mask
    for shot in self.shots:
      if 0 <= shot.x < 100:
        state[int(shot.y // 10), int(shot.x // 10), 3] = True
    state[9, :, 4] = True
    values = (self.steps, self.action, self.multiplier, max(0, self.cooldown),
              round(self.vx * 100), round(self.spawn_distance * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _hit(shot, rock):
  return abs(shot.x - rock.x) < (5 + rock.size) / 2 and abs(shot.y - rock.y) < (9 + rock.size) / 2
