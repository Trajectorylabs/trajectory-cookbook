from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Car:
  x: float
  y: float


class ParkingEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.cars: list[Car] = []
    self.parked: list[Car] = []
    self.angle = -math.pi / 2
    self.car_timer = self.parked_distance = self.road = 0.0
    self.capacity = 1.0
    self.gold: Car | None = None
    self.multiplier = 1
    self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    scroll = 1.05
    if self.cars:
      scroll += max(0, 50 - self.cars[0].y) * 0.1
    else:
      self.car_timer -= 1
    self.road += scroll
    if self.gold is None:
      self.gold = Car(self.random.uniform(40, 70), self.random.uniform(-50, -20))
    self.gold.y += scroll
    if self.gold.y > 103:
      self.gold = None
      self.multiplier = max(1, self.multiplier - 1)
    self.parked_distance -= scroll
    if self.parked_distance < 0:
      self.parked.append(Car(self.random.uniform(80, 84), -5))
      self.parked_distance = (self.random.uniform(8, 12) if self.random.random() < 0.7
                              else self.random.uniform(100, 200))
    for car in self.parked:
      car.y += scroll
    self.parked = [car for car in self.parked if car.y <= 105]
    self.car_timer -= scroll - 1
    if len(self.cars) < self.capacity and self.car_timer < 0:
      self.cars.append(Car(self.random.uniform(20, 60), -5))
      self.car_timer = 15
    self.angle = min(0, max(-math.pi / 2, self.angle + (0.1 if self.pressed else -0.1)))
    reward = 0
    remaining = []
    for car in self.cars:
      heading = -math.pi / 2 if car.y < 3 else self.angle
      car.x += math.cos(heading)
      car.y += scroll + math.sin(heading)
      occupied = (*self.parked, *remaining)
      if any(abs(car.x - other.x) < 4 and abs(car.y - other.y) < 4 for other in occupied):
        self.terminal = True
        break
      if car.x + 2 > 82 and self._parking_strip(car.y):
        reward += self.multiplier * 10
        self.parked.append(car)
        self.capacity += 1 / self.capacity
        continue
      if self.gold is not None and abs(car.x - self.gold.x) < 6 and abs(car.y - self.gold.y) < 6:
        if self.gold.y > -3:
          reward += self.multiplier
          self.multiplier += 1
        self.gold = None
      if car.y > 103:
        self.terminal = True
        break
      remaining.append(car)
    self.cars = remaining
    return reward

  def _parking_strip(self, y):
    return any(start - 2 < y < start + 92 for start in (self.road % 93 + 6, self.road % 93 - 88))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, cars in ((0, self.cars), (1, self.parked), (2, [self.gold] if self.gold else [])):
      for car in cars:
        if 0 <= car.x < 100 and 0 <= car.y < 100:
          state[int(car.y // 10), int(car.x // 10), channel] = True
    for row in range(10):
      state[row, 8, 3] = self._parking_strip(10 * row + 5)
    state[:, (0, 9), 4] = True
    values = (self.steps, self.multiplier, round(-self.angle * 100), round(self.capacity * 20))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, value) & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
