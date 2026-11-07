from dataclasses import dataclass
from math import atan2, cos, hypot, sin

import numpy as np


@dataclass
class Face:
  x: float
  timer: float
  smile: bool
  lead: bool
  target_x: float
  y: float = 20
  target_y: float = 20
  interval: float = 30
  speed: float = 1


@dataclass
class Projectile:
  x: float
  y: float
  vx: float
  vy: float
  bonus: bool


class SmilyAngryEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 50.
    self.direction = self.multiplier = 1
    self.steps = 0
    self.held = self.terminal = False
    self.faces = [Face(30, 20, True, True, 30), Face(70, 50, False, False, 70)]
    self.projectiles = []

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

  def _fire(self, face):
    dx, dy = self.x - face.x, 90 - face.y
    if face.lead:
      vx = self.direction * 0.5
      first = hypot(dx, dy) / face.speed
      second = hypot(dx + vx * first, dy) / face.speed
      dx += vx * second
    angle = atan2(dy, dx)
    self.projectiles.append(Projectile(face.x, face.y, cos(angle) * face.speed,
                                       sin(angle) * face.speed, face.smile))

  def _advance(self, tap):
    for face in self.faces:
      face.x += (face.target_x - face.x) * 0.1
      face.y += (face.target_y - face.y) * 0.1
      face.timer -= 1
      if face.timer < 0:
        self._fire(face)
        face.timer = face.interval
      if self.random.random() < 0.002:
        face.target_x = self.random.uniform(10, 90)
        face.target_y = self.random.uniform(10, 40)
      if self.random.random() < 0.002:
        face.smile = not face.smile
      if self.random.random() < 0.002:
        face.lead = not face.lead
      if self.random.random() < 0.002:
        face.interval = self.random.uniform(30, 40)
        face.speed = self.random.uniform(0.9, 1.2)
    if tap:
      self.direction *= -1
    self.x = (self.x + self.direction * 0.5 + 3) % 106 - 3
    reward = 0
    remaining = []
    for projectile in self.projectiles:
      projectile.x = (projectile.x + projectile.vx + 3) % 106 - 3
      projectile.y += projectile.vy
      reach = 5.5 if projectile.bonus else 4
      if abs(projectile.x - self.x) < reach and abs(projectile.y - 90) < reach:
        if projectile.bonus:
          reward += self.multiplier
          self.multiplier += 1
          continue
        self.terminal = True
      if projectile.y < 93 - (3 if projectile.bonus else 1):
        remaining.append(projectile)
      elif projectile.bonus:
        self.multiplier = max(1, self.multiplier - 1)
    self.projectiles = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(np.clip(self.x // 10, 0, 9)), 0] = True
    for face in self.faces:
      state[int(face.y // 10), int(face.x // 10), 1 if face.smile else 2] = True
    for projectile in self.projectiles:
      if 0 <= projectile.x < 100 and 0 <= projectile.y < 100:
        state[int(projectile.y // 10), int(projectile.x // 10), 3 if projectile.bonus else 4] = True
    values = (self.steps, self.multiplier, self.direction > 0, self.held,
              *(face.lead for face in self.faces), *(round(face.timer) for face in self.faces))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
