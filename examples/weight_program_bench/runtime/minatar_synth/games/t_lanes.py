from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Car:
  x: float
  y: float
  speed: float = 0
  turn: int = 0
  target: float = 0
  cooldown: int = 0
  junction: object = None


@dataclass(eq=False)
class Junction:
  x: float
  y: float
  turn: int
  enabled: bool


class TLanesEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.cars = [Car(20, 50, 0.1), Car(99, 50)]
    self.junctions, self.stops, self.routes = [], [], []
    self.scroll = self.car_dist = self.cross_dist = 0.
    self.stop_dist = 199.
    self.steps = 0
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
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

  def _advance(self, tap):
    moving = sum(car.speed > 0 for car in self.cars)
    if not moving:
      self.terminal = True
      return 0
    reward = 0
    scroll = self.scroll
    self.cross_dist -= scroll
    if self.cross_dist < 0:
      reward += moving
      if not self.routes:
        self.routes = list(self.random.permutation((0, 1, 2, 3, 6, 7, 8, 9)))
      route = self.routes.pop()
      self.junctions.append(Junction(200, 10 + 20 * (route % 5),
                                     1 if route < 5 else -1, self.random.random() < 0.5))
      self.cross_dist = 24
    for junction in self.junctions:
      junction.x -= scroll
      if tap:
        junction.enabled = not junction.enabled
    self.junctions = [junction for junction in self.junctions if junction.x >= -20]
    self.stop_dist -= scroll
    if self.stop_dist < 0:
      self.stops.append((205, 10 + 20 * self.random.integers(5)))
      self.stop_dist += self.random.uniform(99, 120)
    self.stops = [(x - scroll, y) for x, y in self.stops if x - scroll >= -3 and not any(
      y == junction.y and abs(x - scroll - junction.x - 3) < 5 for junction in self.junctions)]
    self.car_dist -= scroll
    if self.car_dist < 0:
      self.cars.append(Car(203, 10 + 20 * self.random.integers(5)))
      self.car_dist += self.random.uniform(40, 60)
    survivors = []
    for car in self.cars:
      car.x -= scroll
      if tap and car.x < 50:
        car.x -= (50 - car.x) * 0.1
      if car.speed:
        car.speed += (1 - car.speed) * 0.05
      if car.turn:
        car.x += car.speed / sqrt(2)
        car.y += car.turn * car.speed / sqrt(2)
        if (car.y - car.target) * car.turn > 0:
          car.y, car.turn, car.cooldown = car.target, 0, 5
      else:
        car.x += car.speed
      car.cooldown -= 1
      if not car.turn and car.cooldown < 0 and any(
        y == car.y and abs(x - car.x) < 5 for x, y in self.stops
      ):
        continue
      if not car.speed and any(other.speed > 0 and not other.turn and
                               abs(other.x - scroll - car.x) < 5 and other.y == car.y
                               for other in self.cars if other is not car):
        reward += moving * 10
        car.speed = self.random.uniform(0.01, 0.3)
      if car.speed and not car.turn:
        junction = next((j for j in self.junctions if j.y == car.y and abs(j.x + 3 - car.x) < 5), None)
        if junction is not None and junction is not car.junction and junction.enabled:
          car.turn = junction.turn
          car.target = car.y + 20 * car.turn
          car.x = junction.x - 2
        car.junction = junction
      if car.x >= -3:
        survivors.append(car)
    self.cars = survivors
    self.terminal = not self.cars
    front = max((car.x for car in self.cars if car.speed), default=0)
    self.scroll = max(0, (front - 50) * 0.1)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for car in self.cars:
      if 0 <= car.x < 200:
        state[int(car.y // 10), int(car.x // 20), int(not car.speed)] = True
    for junction in self.junctions:
      if 0 <= junction.x + 3 < 200:
        state[int(junction.y // 10), int((junction.x + 3) // 20),
              2 + int(junction.enabled)] = True
    for x, y in self.stops:
      if 0 <= x < 200:
        state[int(y // 10), int(x // 20), 4] = True
    for row, value in enumerate((self.steps, self.held, round(self.scroll * 100),
                                 sum(car.speed > 0 for car in self.cars))):
      for bit in range(9):
        state[2 * row + 1, bit, 5] = bool(value & (1 << bit))
    for junction in self.junctions:
      if 0 <= junction.x + 3 < 200 and junction.turn > 0:
        state[max(0, int(junction.y // 10) - 1), int((junction.x + 3) // 20), 5] = True
    return state
