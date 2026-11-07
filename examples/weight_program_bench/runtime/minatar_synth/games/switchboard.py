from dataclasses import dataclass
from math import cos, floor, pi, sin

import numpy as np


SPACING = 2 * pi / 9
PLUGS = tuple((50 + 30 * sin(i * SPACING), 50 - 30 * cos(i * SPACING)) for i in range(9))


@dataclass
class Call:
  source: int
  destination: int
  remaining: int


class SwitchboardEnvironment:
  score_cap = 25

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = 0.
    self.cursor = self.steps = self.lost = 0
    self.origin = None
    self.timer = 180
    self.multiplier = 1.
    self.active = set(range(9))
    self.requests = [Call(2, 6, 900)]
    self.connections = []
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    press, release = held and not self.held, self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(press and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self):
    available = set(range(9))
    for call in (*self.requests, *self.connections):
      available.difference_update((call.source, call.destination))
    sources = available & self.active
    if sources and len(available) >= 2:
      source = int(self.random.choice(sorted(sources)))
      destination = int(self.random.choice(sorted(available - {source})))
      self.requests.append(Call(source, destination, int(self.random.integers(600, 1200))))

  def _lose(self, call):
    self.active.discard(call.source)
    self.lost += 1
    self.terminal |= self.lost >= 3

  def _advance(self, press, release):
    self.angle += 0.024 if self.held else 0.042
    if self.angle >= 2 * pi:
      self.angle = 0.
    self.cursor = floor(self.angle / SPACING + 0.5) % 9
    self.multiplier += (1 - self.multiplier) * 0.001
    self.timer -= 1
    if self.timer <= 0:
      self._spawn()
      self.timer = int(self.random.integers(100, 200))
    remaining = []
    for call in self.requests:
      call.remaining -= 1
      if call.remaining <= 0:
        self._lose(call)
      else:
        remaining.append(call)
    self.requests = remaining
    if press and self.origin is None and self.active:
      self.origin = min(sorted(self.active),
                        key=lambda plug: abs((plug * SPACING - self.angle + pi) % (2 * pi) - pi))
    reward = 0
    if release and self.origin is not None:
      endpoints = {self.origin, self.cursor}
      match = next((call for call in self.requests if {call.source, call.destination} == endpoints), None)
      if match is not None:
        self.requests.remove(match)
        self.connections.append(Call(match.source, match.destination, int(self.random.integers(100, 200))))
        self.lost = max(0, self.lost - len(endpoints - self.active))
        self.active.update(endpoints)
        reward = floor(self.multiplier + 0.5)
        self.multiplier += 1
      else:
        remaining = []
        for call in self.requests:
          if call.source in endpoints:
            self._lose(call)
          else:
            remaining.append(call)
        self.requests = remaining
      self.origin = None
    for call in self.connections:
      call.remaining -= 1
    self.connections = [call for call in self.connections if call.remaining > 0]
    return reward

  @staticmethod
  def _mark(state, plug, channel):
    x, y = PLUGS[plug]
    state[int(y // 10), int(x // 10), channel] = True

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    self._mark(state, self.cursor, 0)
    for plug in self.active:
      self._mark(state, plug, 1)
    if self.origin is not None:
      self._mark(state, self.origin, 2)
    for call in self.requests:
      state[call.source, call.destination, 3] = True
    for call in self.connections:
      state[call.source, call.destination, 4] = True
    values = (self.steps, self.lost, self.held, round(self.multiplier * 32), round(self.angle * 100),
              min((call.remaining for call in self.requests), default=0))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[6, 0, 5] = values[-1] >= 1024
    return state
