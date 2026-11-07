from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float = 0
  vy: float = 0

  def accelerate(self, angle: float, amount: float) -> None:
    self.vx += math.cos(angle) * amount
    self.vy += math.sin(angle) * amount

  def move(self, scale: float = 1) -> None:
    self.x += self.vx * scale
    self.y += self.vy * scale

  def drag(self, scale: float) -> None:
    self.vx *= scale
    self.vy *= scale

  def toward_earth(self) -> float:
    return math.atan2(50 - self.y, 50 - self.x)


def wrap_angle(angle: float) -> float:
  return (angle + math.pi) % math.tau - math.pi


class EarockEnvironment:
  score_cap = 15

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.combo = 1
    self.pressed = self.terminal = False
    self.angle = 0.0
    self.ship = Body(50, 20, 0.1)
    self.stars: list[Body] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    changed = pressed != self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance_frame(changed and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _turn(self, speed: float) -> None:
    offset = wrap_angle(self.ship.toward_earth() - self.angle)
    self.angle = wrap_angle(self.angle + min(speed, max(-speed, offset)))

  def _reflect(self, normal: float) -> None:
    offset = wrap_angle(self.angle + math.pi - normal)
    if abs(offset) < math.pi / 2:
      self.angle = wrap_angle(self.angle + math.pi - 2 * offset)
    speed = math.hypot(self.ship.vx, self.ship.vy) / 2
    self.ship.vx = self.ship.vy = 0
    self.ship.accelerate(self.angle, speed)
    self.combo = 1

  def _advance_frame(self, changed: bool) -> int:
    ship = self.ship
    if self.pressed:
      if changed:
        ship.drag(0.5)
        ship.accelerate(self.angle, 0.1)
      ship.accelerate(self.angle, 0.01)
    else:
      self._turn(0.01)
      if changed:
        self._turn(0.2)
    ship.drag(0.98)
    ship.move()
    if ship.x < 0 or ship.x > 99:
      self._reflect(0 if ship.x < 0 else math.pi)
    if ship.y < 0 or ship.y > 99:
      self._reflect(math.pi / 2 if ship.y < 0 else -math.pi / 2)
    if math.hypot(ship.x - 50, ship.y - 50) < 16.5:
      self._reflect(ship.toward_earth() + math.pi)

    self.spawn_timer = self.spawn_timer - 1 if self.stars else -1
    if self.spawn_timer < 0:
      angle = self.random.uniform(0, math.tau)
      star = Body(50 + 70 * math.cos(angle), 50 + 70 * math.sin(angle))
      star.accelerate(angle + math.pi + self.random.uniform(-1, 1), 0.1)
      self.stars.append(star)
      self.spawn_timer = 300

    reward = 0
    remaining = []
    for star in self.stars:
      star.move()
      star.drag(0.99)
      star.accelerate(star.toward_earth(), 0.0002)
      if math.hypot(star.x - ship.x, star.y - ship.y) < 5:
        reward += self.combo
        self.combo += 1
        continue
      if not (-3 <= star.x < 100 and -3 <= star.y < 100):
        star.move(9)
      if -50 <= star.x < 100 and -50 <= star.y < 100:
        self.terminal |= math.hypot(star.x - 50, star.y - 50) < 15
        remaining.append(star)
    self.stars = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    ship = self.ship
    points = ((ship.x, ship.y, 0),
              (ship.x + 8 * math.cos(self.angle), ship.y + 8 * math.sin(self.angle), 1))
    for x, y, channel in points:
      state[min(9, max(0, int(y) // 10)), min(9, max(0, int(x) // 10)), channel] = True
    for row in range(10):
      for col in range(10):
        state[row, col, 2] = math.hypot(col * 10 + 5 - 50, row * 10 + 5 - 50) < 16.5
    for star in self.stars:
      if 0 <= star.x < 100 and 0 <= star.y < 100:
        state[int(star.y) // 10, int(star.x) // 10, 3] = True
      else:
        state[min(9, max(0, int(star.y) // 10)), min(9, max(0, int(star.x) // 10)), 4] = True
    for row, value in enumerate((self.steps, self.combo, round((self.angle + math.pi) * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
