from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np


CENTER = np.array([50, 60])


@dataclass
class Stone:
  position: np.ndarray
  velocity: np.ndarray
  side: int
  fuse: int = 60


@dataclass
class Ring:
  radius: float
  angle: float
  spread: float
  growth: float


class RingBlastEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.stones = []
    self.rings = []
    self.angle, self.enemy_angle = -pi / 2, pi / 2
    self.speed = self.enemy_speed = 1.0
    self.angle_direction = self.speed_direction = 1
    self.ammo, self.timer = 10, 100
    self.multiplier = 1
    self.action = self.steps = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _aim(self):
    self.angle += 0.03 * ((self.action == 3) - (self.action == 1))
    self.angle = min(-pi / 6, max(-5 * pi / 6, self.angle))
    self.speed = min(4, max(1, self.speed + 0.05 * ((self.action == 2) - (self.action == 4))))
    self.enemy_angle += self.angle_direction * self.random.uniform(0.02, 0.03)
    if abs(self.enemy_angle - pi / 2) > pi / 3 and (self.enemy_angle - pi / 2) * self.angle_direction > 0:
      self.angle_direction *= -1
    self.enemy_speed += self.speed_direction * self.random.uniform(0.03, 0.04)
    if (self.enemy_speed > 4 and self.speed_direction > 0) or (self.enemy_speed < 1 and self.speed_direction < 0):
      self.speed_direction *= -1

  def _advance(self):
    self._aim()
    self.timer = max(0, self.timer - 1)
    if self.timer == 0:
      if self.ammo:
        self.ammo -= 1
        self.timer = 100
        launches = ((120, self.angle, self.speed), (0, self.enemy_angle, self.enemy_speed))
        for side, (y, angle, speed) in enumerate(launches):
          self.stones.append(Stone(np.array([50., float(y)]),
                                   speed * np.array([cos(angle), sin(angle)]), side))
      elif not self.rings and not any(stone.side == 2 for stone in self.stones):
        self.terminal = True
        return 0
    for ring in self.rings:
      ring.spread += ring.growth
    active_rings = tuple(self.rings)
    reward = 0
    survivors = []
    for stone in self.stones:
      stone.position += stone.velocity
      stone.velocity *= 0.98
      lower, upper = np.array([13, 3]), np.array([87, 117])
      bounce = ((stone.position < lower) & (stone.velocity < 0)) | (
        (stone.position > upper) & (stone.velocity > 0)
      )
      stone.velocity[bounce] *= -1
      delta = stone.position - CENTER
      radius = np.linalg.norm(delta)
      angle = atan2(delta[1], delta[0])
      if stone.side == 2:
        stone.fuse -= 1
        if stone.fuse == 0:
          self.rings.append(Ring(radius, angle, 0, 1 / (radius + 1)))
          continue
      elif any(self._ring_hit(ring, radius, angle) for ring in active_rings):
        self.ammo += int(stone.side == 0)
        reward += self.multiplier
        self.multiplier += 1
        continue
      survivors.append(stone)
    for index, stone in enumerate(survivors):
      for other in survivors[index + 1:]:
        delta = other.position - stone.position
        distance = np.linalg.norm(delta)
        if distance >= 6:
          continue
        axis = delta / distance if distance else np.array([1., 0.])
        impulse = 0.7 * (abs(other.velocity @ axis) - abs(stone.velocity @ axis))
        stone.velocity -= impulse * axis
        other.velocity += impulse * axis
        stone.position += 2 * stone.velocity
        other.position += 2 * other.velocity
        if {stone.side, other.side} == {0, 1}:
          (stone if stone.side == 1 else other).side = 2
    self.stones = survivors
    for ring in active_rings:
      ring.radius += 2
    self.rings = [ring for ring in self.rings if ring.radius <= 99]
    if not self.rings:
      self.multiplier = 1
    return reward

  @staticmethod
  def _ring_hit(ring, radius, angle):
    difference = abs((angle - ring.angle + pi) % (2 * pi) - pi)
    tolerance = 4.5 / max(1, ring.radius)
    return abs(radius - ring.radius) < 4.5 and difference <= ring.spread + tolerance

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for stone in self.stones:
      x, y = stone.position
      if 0 <= x < 100 and 0 <= y < 150:
        state[int(y // 15), int(x // 10), stone.side] = True
    rows, cols = np.ogrid[7.5:150:15, 5:100:10]
    radius = np.hypot(cols - 50, rows - 60)
    angle = np.arctan2(rows - 60, cols - 50)
    for ring in self.rings:
      difference = abs((angle - ring.angle + pi) % (2 * pi) - pi)
      state[:, :, 3] |= (abs(radius - ring.radius) < 9) & (
        difference <= ring.spread + 9 / max(1, ring.radius)
      )
    x = 50 + self.speed * 5 * cos(self.angle)
    y = 120 + self.speed * 5 * sin(self.angle)
    state[int(y // 15), int(x // 10), 4] = True
    values = (self.steps, self.ammo, self.timer, self.multiplier,
              round((self.angle + pi) * 100), round(self.speed * 100),
              round(self.enemy_angle * 100), round(self.enemy_speed * 100))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
