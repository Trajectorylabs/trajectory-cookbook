from dataclasses import dataclass

import numpy as np


@dataclass
class Number:
  value: int
  x: float = -3


class NumberLineEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lines = [12.0, 94.0]
    self.targets = self.lines.copy()
    self.numbers: list[list[Number]] = [[], []]
    self.timers = [0.0, 0.0]
    self.sums = [0, 0]
    self.countdown = self.steps = 0
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
    self.targets[0] += 0.01
    self.targets[1] -= 0.01
    for index in range(2):
      self.lines[index] += (self.targets[index] - self.lines[index]) * 0.2
    reward = 0
    if self.lines[0] < 12:
      self.targets[0] += 1
      reward += 1
    if self.lines[1] > 94:
      self.targets[1] -= 1
    self.terminal = self.lines[0] >= self.lines[1]
    if self.countdown:
      self.countdown -= 1
      if not self.countdown:
        difference = self.sums[0] - self.sums[1]
        reward += difference
        self.targets[0] -= difference
        self.targets[1] += difference
        self.sums = [0, 0]
    elif tap:
      for index, numbers in enumerate(self.numbers):
        self.sums[index] = sum(n.value for n in numbers if 3 < n.x < 97)
        self.numbers[index] = [n for n in numbers if not 3 < n.x < 97]
      self.countdown = 61
    for index in range(2):
      self.timers[index] -= 1
      if self.timers[index] < 0:
        self.numbers[index].append(Number(int(self.random.integers(1, 10))))
        self.timers[index] = self.random.uniform(10, 30)
      for number in self.numbers[index]:
        number.x += 0.7
      self.numbers[index] = [n for n in self.numbers[index] if n.x <= 103]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for side, numbers in enumerate(self.numbers):
      for index, number in enumerate(numbers):
        row, col = 4 * side + index // 10 * 2, index % 10
        for bit in range(4):
          state[row, col, bit] = bool(number.value & (1 << bit))
        position = round(number.x + 3)
        for bit in range(7):
          state[row + (bit >= 2), col, bit + 4 if bit < 2 else bit - 2] = bool(position & (1 << bit))
    values = (self.steps, self.countdown, self.sums[0], self.sums[1],
              round(self.lines[0]) + 128, round(self.lines[1]) + 128)
    for channel, value in enumerate(values):
      for bit in range(9):
        state[8, bit, channel] = bool(value & (1 << bit))
    state[9, 0, 0] = self.pressed
    return state
