from dataclasses import dataclass
from math import atan2

import numpy as np


@dataclass
class Car:
  x: float
  size: float
  target: float
  speed: float = 0.
  angle: float = 0.


class UpDownPressEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.road = [(0., 50.), (100., 50.)]
    self.road_distance, self.car_distance = -250., 0.
    self.cars = []
    self.x, self.y, self.speed = 20., 50., 1.
    self.vx = self.vy = self.angle = 0.
    self.airborne = self.terminal = False
    self.multiplier = 1
    self.steps = self.action = 0

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    jump = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(jump and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _surface(self, x):
    for (left, y), (right, end_y) in zip(self.road, self.road[1:]):
      if x < right:
        return y + (x - left) * (end_y - y) / (right - left), atan2(end_y - y, right - left)
    return self.road[-1][1], 0.

  def _advance(self, jump):
    scroll = 0.1 + max(0, self.x - 50) * 0.1
    ahead, _ = self._surface(self.x + 50)
    vertical = 0.1 + (ahead - min(90, max(60, ahead))) * 0.1
    self.road_distance -= scroll
    while self.road_distance < 0:
      x, y = self.road[-1]
      width = self.random.uniform(20, 60)
      rise = 0.
      if y == self.road[-2][1]:
        rise = self.random.choice((-1, 1)) * self.random.uniform(0.4, 1.1) * width
      self.road.append((x + width, y + rise))
      self.road_distance += width
    self.road = [(x - scroll, y - vertical) for x, y in self.road]
    while len(self.road) > 2 and self.road[1][0] < -50:
      self.road.pop(0)

    self.x += self.speed - scroll + self.vx
    self.vx *= 0.9
    if self.x < 0:
      self.terminal = True
      return 0
    height, slope = self._surface(self.x)
    if self.airborne and self.y > height:
      self.airborne = False
    if not self.airborne:
      self.y = height
      self.speed += self.angle * 0.1
      self.angle += (slope - self.angle) * 0.025
      if jump:
        self.airborne = True
        self.vy = -2
        self.multiplier = 1
    if self.airborne:
      self.y += self.vy
      self.vy += 0.05 if self.action == 5 else 0.2
      self.angle += (atan2(self.vy, self.speed) - self.angle) * 0.05
    self.speed += ((2.5 if self.action == 5 else 0.5) - self.speed) * 0.1
    return self._traffic(scroll)

  def _traffic(self, scroll):
    self.car_distance -= scroll
    if self.car_distance < 0:
      target = self.random.uniform(0.3, 2)
      self.cars.append(Car(205, self.random.uniform(5, 8), target))
      self.car_distance += self.random.uniform(100, 120)
    reward = 0
    remaining = []
    for car in self.cars:
      car.x += car.speed - scroll
      if not self.road[0][0] <= car.x < self.road[-1][0]:
        continue
      y, slope = self._surface(car.x)
      car.speed += car.size * car.angle * 0.02
      car.speed += (car.target - car.speed) * 0.1
      car.angle += (slope - car.angle) * 0.025
      if abs(car.x - self.x) < (car.size + 5) / 2:
        pressing = self.airborne and self.vy >= 0
        if pressing and abs((self.y - 3) - (y - car.size)) < 6:
          reward += self.multiplier
          self.multiplier = min(16, self.multiplier * 2)
          self.vy = -2
          continue
        if not pressing and abs((self.y - 5) - (y - car.size)) < 5:
          self.vx = -car.size - self.speed * 1.5
      remaining.append(car)
    self.cars = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    _mark(state, self.x, self.y, 0)
    for x in range(0, 200, 4):
      _mark(state, x, self._surface(x)[0], 1)
    for car in self.cars:
      _mark(state, car.x, self._surface(car.x)[0], 2)
    if self.airborne:
      _mark(state, self.x, self.y, 3 if self.vy >= 0 else 4)
    values = (self.steps, self.action, self.multiplier, round(self.speed * 64),
              round((self.vy + 4) * 32), round((self.vx + 16) * 16),
              round((self.angle + 2) * 64))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _mark(state, x, y, channel):
  if 0 <= x < 200 and 0 <= y < 100:
    state[int(y // 10), int(x // 20), channel] = True
