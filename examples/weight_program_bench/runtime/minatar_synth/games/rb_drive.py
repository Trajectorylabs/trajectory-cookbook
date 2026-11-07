from dataclasses import dataclass

import numpy as np


DIRECTIONS = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)))


@dataclass
class Car:
  lane: int
  color: int
  phase: float = 0
  entry: int = 0

  def position(self):
    direction = DIRECTIONS[self.lane]
    shoulder = DIRECTIONS[(self.lane + 1) % 4] * 5
    if self.phase < 1:
      offset = direction * (50 - 40 * self.phase) + shoulder
    elif self.phase < 2:
      start = DIRECTIONS[self.entry] * 10 + DIRECTIONS[(self.entry + 1) % 4] * 5
      end = direction * 10 - shoulder
      offset = start + (end - start) * (self.phase - 1)
    else:
      offset = direction * (10 + 40 * (self.phase - 2)) - shoulder
    return 50 + offset


class RBDriveEnvironment:
  score_cap = 16

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lane = 3
    self.health = np.full(4, 0.7)
    self.cars = []
    self.spawn_timer = self.color_count = self.steps = 0
    self.color = int(self.random.integers(2))
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.lane = (self.lane + 1) % 4
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
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self.color_count -= 1
      if self.color_count < 0:
        self.color = 1 - self.color
        self.color_count = int(self.random.integers(1, 4))
      self.cars.append(Car(int(self.random.integers(4)), self.color))
      self.spawn_timer += self.random.uniform(80, 99)
    reward = 0
    remaining = []
    for car in self.cars:
      previous = car.phase
      car.phase += 0.05 if 1 < previous < 2 else 0.01
      if previous < 1 <= car.phase:
        car.entry = car.lane
        car.lane = self.lane if self.lane != car.entry else (car.entry + 2) % 4
      if car.phase <= 3:
        remaining.append(car)
      elif car.lane % 2 == car.color:
        self.health[car.lane] *= 0.66
      else:
        self.health[car.lane] = min(1, self.health[car.lane] + 0.3)
        reward += 1
    self.cars = remaining
    self.health -= 0.0006
    self.terminal = bool(np.any(self.health < 0.01))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for car in self.cars:
      x, y = np.clip(car.position() // 10, 0, 9).astype(int)
      state[y, x, car.color] = True
      if car.phase >= 1:
        state[y, x, 4] = True
    for lane, (dx, dy) in enumerate(DIRECTIONS):
      for distance in range(1, 5):
        row, col = 5 + dy * distance, 5 + dx * distance
        state[row, col, 2] = lane == self.lane
        state[row, col, 3] = self.health[lane] > (distance - 1) / 4
    gauges = (self.steps, self.held, *(round(value * 100) for value in self.health))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
