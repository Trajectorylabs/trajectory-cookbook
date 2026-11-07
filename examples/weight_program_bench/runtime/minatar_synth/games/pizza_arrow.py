import math

import numpy as np


class PizzaArrowEnvironment:
  score_cap = 2500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = self.low = self.high = 0.0
    self.y = -25
    self.arrow = None
    self.velocity = self.steps = 0
    self.speed = 1.0
    self.arrows = self.capacity = self.multiplier = 1
    self.timer = 16
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    release = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, release):
    self.timer -= 1
    if self.timer == 15:
      self.angle = self.random.uniform(0, math.pi / 2)
      self.low, self.high = math.pi / 4, 7 * math.pi / 4
      self.multiplier = 1
    if self.timer >= 0:
      self.y = 50 - self.timer * 5 if self.timer <= 15 else 50 + (30 - self.timer) * 5
    cut_angle = -self.angle % math.tau
    self.angle = (self.angle + 0.2 * self.speed) % math.tau
    if self.timer < 0 and self.arrow is None and self.pressed:
      self.arrow, self.velocity = 80.0, 1
      self.arrows -= 1
    if self.arrow is None:
      return 0
    if self.pressed:
      self.speed += (0.05 - self.speed) * 0.1
    if release or self.arrow > 90:
      self.velocity = -5
    if self.velocity < 0:
      self.speed += (1 - self.speed) * 0.2
    self.arrow += self.velocity * self.speed
    radius = self.arrow - 30
    if radius <= 22 and self.low < cut_angle < self.high:
      return self._cut(cut_angle)
    if radius <= 7:
      self.terminal = True
      self.arrow = None
    return 0

  def _cut(self, angle):
    left, right = angle - self.low, self.high - angle
    if left < right:
      self.low = angle
    else:
      self.high = angle
    reward = math.ceil(min(left, right) * 100 * self.multiplier)
    self.multiplier += 1
    self.arrow = None
    if not self.arrows:
      self.capacity += 1
      self.arrows = self.capacity
      self.timer = 30
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for angle in np.linspace(self.low, self.high, 40):
      x, y = 30 + 20 * math.cos(angle + self.angle), self.y + 20 * math.sin(angle + self.angle)
      if 0 <= y < 100:
        state[int(y // 10), int(x // 10), 0] = True
    for angle in (self.low, self.high):
      for radius in range(5, 21, 3):
        x = 30 + radius * math.cos(angle + self.angle)
        y = self.y + radius * math.sin(angle + self.angle)
        if 0 <= y < 100:
          state[int(y // 10), int(x // 10), 1] = True
    if 0 <= self.y < 100:
      state[self.y // 10, 3, 2] = True
    if self.arrow is not None:
      state[5, int(self.arrow // 10), 3] = True
    state[5, :, 4] = True
    values = (self.steps, self.arrows, self.multiplier, round(self.angle * 80),
              round(self.low * 80), round(self.high * 80), round(self.speed * 100))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.velocity < 0 and self.arrow is not None
    return state
