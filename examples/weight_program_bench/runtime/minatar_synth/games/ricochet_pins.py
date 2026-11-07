from dataclasses import dataclass

import numpy as np


@dataclass
class Pin:
  position: np.ndarray
  velocity: np.ndarray


class RicochetPinsEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.position = np.full(2, 50.0)
    self.velocity = np.array([0.5, 0.0])
    self.steps = self.heat = self.shot_timer = 0
    self.pin_count = 5
    self.multiplier = 1
    self.held = self.overheated = self.terminal = False
    self.shot_target = None
    self._spawn_wave()

  def _spawn_wave(self):
    self.pins = []
    for _ in range(self.pin_count):
      for _ in range(50):
        position = self.random.uniform(15, 85, 2)
        if np.linalg.norm(position - self.position) > 30:
          break
      velocity = np.zeros(2)
      if self.random.random() < 0.4:
        velocity[self.random.integers(2)] = self.random.choice((-0.35, 0.35))
      self.pins.append(Pin(position, velocity))
    for _ in range(50):
      self.item = self.random.uniform(15, 85, 2)
      if np.linalg.norm(self.item - self.position) > 20 and all(
        np.linalg.norm(self.item - pin.position) >= 15 for pin in self.pins
      ):
        break

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

  @staticmethod
  def _bounce(position, velocity, radius):
    crossed = (position < radius) | (position > 100 - radius)
    velocity[crossed] *= -1
    np.clip(position, radius, 100 - radius, out=position)

  def _advance(self, tap):
    self.position += self.velocity
    self.velocity *= 0.98
    self.heat = max(0, self.heat - 1)
    if self.heat <= 30:
      self.overheated = False
    radius = 3 * (1 + 0.008 * self.heat)
    self._bounce(self.position, self.velocity, radius)
    for pin in self.pins:
      pin.position += pin.velocity
      self._bounce(pin.position, pin.velocity, 3)
    distances = [np.linalg.norm(pin.position - self.position) for pin in self.pins]
    self.terminal = any(distance < radius + 3 for distance in distances)
    reward = 0
    if self.item is not None and np.linalg.norm(self.item - self.position) < 7:
      reward += 10 * self.multiplier
      self.multiplier += 1
      self.item = None
    self.shot_timer = max(0, self.shot_timer - 1)
    if tap and distances and not self.overheated and not self.terminal:
      index = int(np.argmin(distances))
      self.shot_target = self.pins.pop(index).position
      self.shot_timer = 9
      self.velocity += 2 * (self.position - self.shot_target) / distances[index]
      reward += self.multiplier
      self.heat = min(100, self.heat + 25)
      self.overheated = self.heat == 100
      if not self.pins:
        self.pin_count = min(64, self.pin_count + 1)
        self._spawn_wave()
        reward += self.pin_count * self.multiplier * 2
    return reward

  @staticmethod
  def _mark(state, position, channel):
    col, row = (position // 10).astype(int)
    state[row, col, channel] = True

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    self._mark(state, self.position, 0)
    for pin in self.pins:
      self._mark(state, pin.position, 1 + int(np.any(pin.velocity)))
    if self.item is not None:
      self._mark(state, self.item, 3)
    if self.shot_timer:
      for point in np.linspace(self.position, self.shot_target, 20):
        self._mark(state, point, 4)
    gauges = (self.steps, self.heat, self.overheated, self.multiplier,
              self.pin_count, self.held, round(self.velocity[0] * 20) + 256,
              round(self.velocity[1] * 20) + 256)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
