from collections import deque

import numpy as np


class Snake1Environment:
  score_cap = 20
  directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.head = (8, 8)
    self.direction = self.timer = self.steps = 0
    self.rotation = 1
    self.queued = self.grow = self.held = self.terminal = False
    self.body = deque((x, 8) for x in range(4, 8))
    self.dollars = [(12, 8)]

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    self.queued |= held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      self.timer -= 1
      if self.timer < 0:
        self.timer = 20
        reward += self._move()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _move(self):
    if not self.grow:
      self.body.popleft()
    self.grow = False
    self.body.append(self.head)
    if self.queued:
      self.direction = (self.direction + self.rotation) % 4
      self.queued = False
    dx, dy = self.directions[self.direction]
    self.head = self.head[0] + dx, self.head[1] + dy
    x, y = self.head
    self.terminal = self.head in self.body or not (1 <= x <= 14 and 2 <= y <= 14)
    if self.head not in self.dollars:
      return 0
    self.dollars = [dollar for dollar in self.dollars if dollar != self.head]
    self.grow = True
    self.rotation *= -1
    for _ in range(2):
      for _ in range(99):
        position = tuple(self.random.integers((2, 3), (14, 14)))
        if position != self.head and position not in self.body:
          self.dollars.append(position)
          break
    return 1

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)

    def mark(x, y, channel):
      state[min(9, max(0, (y - 1) * 10 // 15)), min(9, max(0, x * 10 // 16)), channel] = True

    mark(*self.head, 0)
    for point in self.body:
      mark(*point, 1)
    for point in self.dollars:
      mark(*point, 2)
    state[[0, 9], :, 3] = True
    state[:, [0, 9], 3] = True
    dx, dy = self.directions[(self.direction + self.rotation) % 4]
    mark(self.head[0] + dx, self.head[1] + dy, 4)
    values = (self.steps, self.direction, self.rotation > 0, self.timer,
              self.queued, self.grow, self.held, *self.head)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
