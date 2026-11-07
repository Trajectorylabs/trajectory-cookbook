from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float = 0
  vy: float = 0

  def move(self, scroll=0):
    self.x += self.vx - scroll
    self.y += self.vy


@dataclass
class Plane(Body):
  fuse: float = 0
  base_fuse: float = 0


class MRiderEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.rider = Body(30, 46)
    self.mount = Body(30, 50, 1)
    self.missiles = [self.mount]
    self.planes: list[Plane] = []
    self.missile_timer, self.plane_timer = 9, 0
    self.steps = 0
    self.multiplier = 1
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
    rider, mount = self.rider, self.mount
    if mount is None:
      rider.vx = (rider.vx + (0.04 if self.pressed else -0.02)) * 0.99
      rider.vy = (rider.vy + (0.01 if self.pressed else 0.05)) * 0.99
      rider.move()
    else:
      mount.vy = (mount.vy + (-0.05 if self.pressed else 0.05)) * 0.99
      mount.vx += (1 - mount.vx) * 0.1
      rider.x, rider.y = mount.x, mount.y - 4
      rider.vx, rider.vy = mount.vx, mount.vy
    scroll = max(0, rider.x - 50) * 0.1
    rider.x -= scroll
    if rider.y > 99 or (mount is not None and rider.y < -1):
      self.terminal = True
      return 0
    self._move_missiles(scroll)
    return self._move_planes(scroll)

  def _move_missiles(self, scroll):
    self.missile_timer -= 1
    if self.missile_timer < 0:
      y = self.random.uniform(40, 70)
      angle = math.atan2(y - self.random.uniform(50, 60), 210)
      speed = self.random.uniform(0.4, 0.5)
      self.missiles.append(Body(210, y, speed * math.cos(angle), speed * math.sin(angle)))
      self.missile_timer = self.random.uniform(60, 300)
    airborne = self.mount is None
    for missile in self.missiles:
      missile.move(scroll)
      if airborne and abs(missile.x - self.rider.x) < 12 and abs(missile.y - self.rider.y) < 5:
        self.multiplier = (self.multiplier + 1) // 2
        self.mount = missile
    self.missiles = [m for m in self.missiles if -9 <= m.x < 221 and -3 <= m.y < 113]

  def _move_planes(self, scroll):
    self.plane_timer -= 1
    if self.plane_timer < 0:
      y = self.random.uniform(30, 90)
      angle = math.atan2(y - self.random.uniform(40, 90), 203)
      speed = self.random.uniform(0.05, 0.3)
      self.planes.append(Plane(203, y, speed * math.cos(angle), speed * math.sin(angle)))
      self.plane_timer = self.random.uniform(20, 24)
    reward = 0
    remaining = []
    for plane in self.planes:
      plane.move(scroll)
      if plane.fuse > 0:
        plane.fuse -= 1
        if plane.fuse <= 0:
          reward += self.multiplier
          self.multiplier += 1
          delay = plane.base_fuse * 0.9
          if delay > 2:
            for other in self.planes:
              if other is not plane and math.hypot(other.x - plane.x, other.y - plane.y) < 36:
                other.fuse = other.base_fuse = delay
          continue
      if self.mount is not None and abs(self.mount.x - plane.x) < 12 and abs(self.mount.y - plane.y) < 5:
        self.rider.vy = -2
        self.mount.x = -99
        self.mount = None
        plane.fuse = plane.base_fuse = 9
      if -3 <= plane.x < 203 and -3 <= plane.y < 103:
        remaining.append(plane)
    self.planes = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rider = self.rider
    state[min(9, max(0, int(rider.y // 10))), min(9, max(0, int(rider.x // 20))), 0] = True
    for missile in self.missiles:
      if 0 <= missile.x < 200 and 0 <= missile.y < 100:
        state[int(missile.y // 10), int(missile.x // 20), 2 if missile is self.mount else 1] = True
    for plane in self.planes:
      if 0 <= plane.x < 200 and 0 <= plane.y < 100:
        state[int(plane.y // 10), int(plane.x // 20), 4 if plane.fuse else 3] = True
    for row, value in enumerate((self.steps, self.multiplier, round((rider.vy + 6) * 35),
                                 round((rider.vx + 4) * 50))):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, :2, 5] = self.pressed, self.mount is not None
    return state
