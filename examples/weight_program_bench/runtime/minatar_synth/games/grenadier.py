from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Tank:
  x: float
  timer: float
  interval: float


@dataclass
class Grenade:
  x: float
  y: float
  vx: float
  vy: float


class GrenadierEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = 10.0
    self.mode = "cover"
    self.angle = 0.0
    self.speed = 1.0
    self.spawn_timer = 0.0
    self.steps = 0
    self.pressed = self.terminal = False
    self.holes = [10.0]
    self.tanks: list[Tank] = []
    self.bullets: list[float] = []
    self.grenades: list[Grenade] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap, release = pressed and not self.pressed, self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for frame in range(4):
      reward += self._advance(tap and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, tap: bool, release: bool) -> float:
    scroll = (self.x - 10) * 0.05
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      interval = 300 / (1 + self.random.random())
      self.tanks.append(Tank(203, self.random.uniform(0, interval), interval))
      self.spawn_timer = 200 / (1 + self.random.random())
    for tank in self.tanks:
      tank.x -= 0.08 * self.speed + scroll
    self.holes = [x - scroll for x in self.holes if x - scroll > -4]
    reward = 0.0
    airborne = []
    for grenade in self.grenades:
      grenade.x += grenade.vx
      grenade.y += grenade.vy
      grenade.vy += 0.1
      hit = next((tank for tank in self.tanks
                  if abs(tank.x - grenade.x) < 4 and abs(67 - grenade.y) < 4), None)
      if hit is not None:
        reward += max(0, hit.x)
        self.tanks.remove(hit)
      elif grenade.y > 68:
        self.holes.append(grenade.x)
      else:
        airborne.append(grenade)
    self.grenades = airborne
    self.tanks = [tank for tank in self.tanks if tank.x > -4]
    if not self.tanks:
      self.spawn_timer *= 0.7
    for tank in self.tanks:
      tank.timer -= 1
      if tank.x > 150 and self.mode != "cover" and tank.timer < 0:
        self.bullets.append(tank.x - 5)
        tank.timer = tank.interval
    self.bullets = [x - 0.4 * self.speed - scroll for x in self.bullets
                    if x - 0.4 * self.speed - scroll > -4]
    self.holes = [x for x in self.holes if all(abs(tank.x - x) >= 6 for tank in self.tanks)]
    self.x -= scroll
    if self.mode == "cover":
      self.speed += 0.05
      if tap:
        self.mode = "run"
        self.x += 6
      else:
        self.terminal = any(abs(tank.x - self.x) < 5 for tank in self.tanks)
        if any(abs(x - self.x - 5) < 6 for x in self.holes):
          self.x += 1
    else:
      self.speed += (1 - self.speed) * 0.1
      if self.mode == "run":
        self.x += 0.8
        if any(abs(x - self.x) < 5 for x in self.holes):
          self.mode = "cover"
        elif tap:
          self.mode = "aim"
          self.angle = 0.0
      elif release or self.angle < -1:
        speed = 4 - self.angle * 0.5
        self.grenades.append(Grenade(self.x, 67, speed * math.cos(self.angle),
                                    speed * math.sin(self.angle)))
        self.mode = "run"
      else:
        self.angle -= 0.02
      self.terminal = any(abs(tank.x - self.x) < 5 for tank in self.tanks) or any(
        abs(x - self.x) < 4 for x in self.bullets
      )
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9 if self.mode == "cover" else 8, int(self.x) // 20, 0] = True
    for channel, positions in ((1, self.holes), (2, [tank.x for tank in self.tanks]), (3, self.bullets)):
      for x in positions:
        if 0 <= x < 200:
          state[8, int(x) // 20, channel] = True
    for grenade in self.grenades:
      if 0 <= grenade.x < 200 and 0 <= grenade.y < 80:
        state[int(grenade.y) // 8, int(grenade.x) // 20, 4] = True
    for row, value in enumerate((self.steps, round(-100 * self.angle))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 5] = self.pressed, self.mode == "cover", self.mode == "aim"
    return state
