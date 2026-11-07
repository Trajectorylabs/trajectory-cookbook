from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np


CENTER = np.array([50., 50.])


@dataclass
class Rock:
  position: np.ndarray
  velocity: np.ndarray
  size: float


@dataclass
class Geode:
  position: np.ndarray
  velocity: np.ndarray
  age: int


class TectonicTetherEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.position = np.array([50., 10.])
    self.angle, self.angular_speed, self.length = 0., 0.01, 35.
    self.charge = self.boost = 0.
    self.power = 30.
    self.steps = 0
    self.held = self.terminal = False
    self.rocks = [Rock(self.random.uniform(20, 80, 2), np.zeros(2),
                       self.random.uniform(8, 12)) for _ in range(5)]
    self.geodes = [Geode(self.random.uniform(10, 90, 2), np.zeros(2), 60) for _ in range(7)]

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if self.held and not held:
      self.boost, self.charge = self.charge, 0.
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
    if self.held:
      self.length -= 0.7
      self.angular_speed = min(0.05, self.angular_speed + 0.001)
      if self.length <= 10:
        self.charge = min(5, self.charge + 0.1)
    else:
      self.length += 0.7 + self.boost
      self.angular_speed = max(0.01, self.angular_speed - 0.0005)
      self.boost *= 0.92
    self.angle = (self.angle + self.angular_speed) % (2 * pi)
    self.position = np.clip(CENTER + self.length * np.array([cos(self.angle), sin(self.angle)]), 3, 97)
    self.length = max(10, float(np.linalg.norm(self.position - CENTER)))
    self.power -= 0.05
    self.terminal = self.power <= 0
    reward = 0
    remaining = []
    for geode in self.geodes:
      geode.age = min(60, geode.age + 1)
      if geode.age < 60:
        geode.position += geode.velocity
        geode.velocity *= 0.96
      else:
        delta = self.position - geode.position
        distance = np.linalg.norm(delta)
        if 0 < distance < 15 + self.power:
          geode.position += delta / distance * (0.3 + self.power * 0.05)
      np.clip(geode.position, 5, 95, out=geode.position)
      if geode.age == 60 and np.all(abs(geode.position - self.position) < 5.5):
        self.power = min(30, self.power + 1.5)
        reward += 1
      else:
        remaining.append(geode)
    self.geodes = remaining
    for index, rock in enumerate(self.rocks):
      rock.position += rock.velocity
      rock.velocity *= 0.95
      delta = rock.position - self.position
      if np.all(abs(delta) < rock.size / 2 + 3):
        count = int(self.angular_speed * self.length)
        outside = np.linalg.norm(rock.position - CENTER) > np.linalg.norm(self.position - CENTER)
        self.length += -5 if outside else 5
        distance = np.linalg.norm(delta)
        if distance:
          rock.velocity += delta / distance * 0.8
        angle = atan2(-delta[1], -delta[0])
        self._scatter(rock, count, angle, pi / 3, 3, 0.45, 1.05)
      edge = rock.size / 2 + 2
      if np.any((rock.position < edge) | (rock.position > 100 - edge)):
        reward += 10
        self._scatter(rock, int(rock.size / 2 + 1), pi, pi, 8, 1, 1.5)
        rock.position = self.random.uniform(20, 80, 2)
        rock.velocity[:] = 0
      for other in self.rocks[index + 1:]:
        offset = rock.position - other.position
        distance = np.linalg.norm(offset)
        limit = (rock.size + other.size) / 2
        if 0 < distance < limit:
          normal = offset / distance
          separation = normal * (limit - distance) / 2
          rock.position += separation
          other.position -= separation
          rock.velocity += normal * 0.4
          other.velocity -= normal * 0.4
    return reward

  def _scatter(self, rock, count, angle, spread, jitter, low, high):
    for _ in range(count):
      heading = angle + self.random.uniform(-spread, spread)
      position = np.clip(rock.position + self.random.uniform(-jitter, jitter, 2), 10, 90)
      speed = self.random.uniform(low, high)
      self.geodes.append(Geode(position, speed * np.array([cos(heading), sin(heading)]), 0))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = (self.position // 10).astype(int)
    state[y, x, 0] = True
    for rock in self.rocks:
      x, y = np.clip(rock.position // 10, 0, 9).astype(int)
      state[y, x, 1] = True
    for geode in self.geodes:
      x, y = (geode.position // 10).astype(int)
      state[y, x, 2 if geode.age == 60 else 3] = True
    cols, rows = (np.linspace(CENTER, self.position, 20) // 10).astype(int).T
    state[rows, cols, 4] = True
    values = (self.steps, round(max(0, self.power) * 10), round(self.charge * 20),
              self.held, round(self.length * 5), round(self.angular_speed * 1000))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
