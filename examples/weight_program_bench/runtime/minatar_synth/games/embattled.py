from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Tank:
  x: float
  y: float
  side: int
  heading: float
  turret: float
  timer: float
  interval: float


@dataclass
class Bullet:
  x: float
  y: float
  angle: float
  side: int


def turn(angle: float, target: float, rate: float) -> float:
  offset = (target - angle + math.pi) % math.tau - math.pi
  return angle + min(rate, max(-rate, offset))


class EmbattledEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = self.held = self.side = self.side_count = 0
    self.pressed = self.terminal = False
    self.y, self.direction, self.multiplier = 30.0, 1, 1
    self.spawn_timer = 0.0
    self.tanks: list[Tank] = []
    self.bullets: list[Bullet] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
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

  def _advance(self, tap: bool) -> int:
    if tap or self.y < 3 and self.direction < 0 or self.y > 97 and self.direction > 0:
      self.direction *= -1
    self.held = self.held + 1 if self.pressed else 0
    if not self.pressed:
      self.y += self.direction * 0.5
    shield = self.held > 10
    pace = 2 if shield else 1
    live_bullets = []
    for bullet in self.bullets:
      bullet.x += math.cos(bullet.angle) * 0.5 * pace
      bullet.y += math.sin(bullet.angle) * 0.5 * pace
      dx, dy = abs(bullet.x - 50), abs(bullet.y - self.y)
      if shield and dx < 8 and dy < 9:
        continue
      self.terminal |= dx < 4 and dy < 4
      if -5 <= bullet.x <= 105 and -5 <= bullet.y <= 105:
        live_bullets.append(bullet)
    self.bullets = live_bullets

    self.spawn_timer = self.spawn_timer - pace if self.tanks else -1
    if self.spawn_timer < 0:
      self._spawn()
    for tank in self.tanks:
      self._move_tank(tank, shield, pace)

    reward = 0
    survivors = []
    for tank in self.tanks:
      hit = next((bullet for bullet in self.bullets if bullet.side != tank.side
                  and math.hypot(bullet.x - tank.x, bullet.y - tank.y) < 4), None)
      if hit is not None:
        self.bullets.remove(hit)
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= abs(tank.x - 50) < 4 and abs(tank.y - self.y) < 4
        if -5 <= tank.x <= 105 and -5 <= tank.y <= 105:
          survivors.append(tank)
    self.tanks = survivors
    if self.frame % 60 == 0:
      self.multiplier = max(1, self.multiplier - 1)
    self.frame += 1
    return reward

  def _spawn(self) -> None:
    side = self.side
    self.side_count -= 1
    if self.side_count <= 0:
      self.side = 1 - self.side
      self.side_count = int(self.random.integers(1, 4))
    x, y = (-5 if side == 0 else 105), self.random.uniform(0, 99)
    angle = math.atan2(self.y - y, 50 - x)
    interval = self.random.uniform(300, 400)
    self.tanks.append(Tank(x, y, side, angle, angle, self.random.uniform(0, interval), interval))
    self.spawn_timer = self.random.uniform(60, 80)

  def _move_tank(self, tank: Tank, shield: bool, pace: int) -> None:
    target = (50, self.y) if shield else None
    distance = math.hypot(50 - tank.x, self.y - tank.y) if shield else 99
    for other in self.tanks:
      if tank.side != other.side:
        candidate = math.hypot(other.x - tank.x, other.y - tank.y)
        if candidate < distance:
          target, distance = (other.x, other.y), candidate
    if target is not None:
      angle = math.atan2(target[1] - tank.y, target[0] - tank.x)
      tank.heading = turn(tank.heading, angle, 0.02)
      tank.turret = turn(tank.turret, angle, 0.03 * pace)
    tank.x += math.cos(tank.heading) * 0.02 * pace
    tank.y += math.sin(tank.heading) * 0.02 * pace
    tank.timer -= pace
    if tank.timer < 0:
      self.bullets.append(Bullet(tank.x, tank.y, tank.turret, tank.side))
      tank.timer = tank.interval

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(self.y) // 10
    state[row, 5, 0] = True
    if self.held > 10:
      state[max(0, row - 1):row + 2, 4:6, 1] = True
    for tank in self.tanks:
      if 0 <= tank.x < 100 and 0 <= tank.y < 100:
        state[int(tank.y) // 10, int(tank.x) // 10, 2 + tank.side] = True
    for bullet in self.bullets:
      if 0 <= bullet.x < 100 and 0 <= bullet.y < 100:
        state[int(bullet.y) // 10, int(bullet.x) // 10, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, min(self.held, 11))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
