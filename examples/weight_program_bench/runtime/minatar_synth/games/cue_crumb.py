from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


@dataclass(eq=False)
class Body:
  x: float
  y: float
  vx: float = 0
  vy: float = 0

  @property
  def speed(self) -> float:
    return math.hypot(self.vx, self.vy)

  def move(self, friction: float, margin: float) -> None:
    self.vx *= friction
    self.vy *= friction
    self.x += self.vx
    self.y += self.vy
    if not margin <= self.x <= 100 - margin:
      self.vx *= -1
      self.x = min(100 - margin, max(margin, self.x))
    if not margin <= self.y <= 100 - margin:
      self.vy *= -1
      self.y = min(100 - margin, max(margin, self.y))


@dataclass(eq=False)
class Rock(Body):
  size: float = 9
  hot: int = 0
  depth: int = 0


class CueCrumbEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.freeze = 0
    self.spawn_timer = 300
    self.pressed = self.terminal = False
    self.player = Body(50, 50)
    self.angle = self.random.uniform(0, math.tau)
    self.rocks: list[Rock] = []
    for _ in range(4):
      x, y = self.random.uniform(12, 88, 2)
      if math.hypot(x - 50, y - 50) < 20:
        x = 12 + (x - 12 + 35) % 76
      self.rocks.append(Rock(x, y))

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
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, tap: bool) -> int:
    if not self.freeze:
      self.angle = (self.angle + 0.058) % math.tau
    if tap:
      self.player.vx += 1.7 * math.cos(self.angle)
      self.player.vy += 1.7 * math.sin(self.angle)
    if self.freeze:
      self.freeze -= 1
      return 0

    self.player.move(0.965, 5)
    self._spawn()
    reward = self._collide_rocks()
    for rock in self.rocks[:]:
      rock.hot = max(0, rock.hot - 1)
      rock.move(0.985, 4)
      dx, dy = rock.x - self.player.x, rock.y - self.player.y
      if math.hypot(dx, dy) >= rock.size / 2 + 3:
        continue
      angle = math.atan2(dy, dx)
      speed = self.player.speed
      if rock.size <= 4.5 and not rock.hot:
        self.rocks.remove(rock)
        reward += 3
      elif rock.hot and rock.speed > 0.4:
        self.terminal = True
        break
      elif speed > 0.9 and rock.size > 4.5:
        self._split(rock, angle, 1.2 + speed * 0.4, 1)
        self.player.vx *= 0.3
        self.player.vy *= 0.3
        self.freeze = max(self.freeze, 2 + 2 * (speed > 1.4) + 2 * (speed > 2.2))
        reward += 1
      else:
        rock.vx += math.cos(angle) * 0.5
        rock.vy += math.sin(angle) * 0.5
    return reward

  def _spawn(self) -> None:
    self.spawn_timer -= 1
    if self.spawn_timer >= 0 or sum(rock.size > 4.5 for rock in self.rocks) >= 10:
      return
    side = int(self.random.integers(4))
    offset = self.random.uniform(4, 96)
    x, y = ((offset, 4), (96, offset), (offset, 96), (4, offset))[side]
    self.rocks.append(Rock(x, y, *self.random.uniform(-0.2, 0.2, 2)))
    self.spawn_timer = int(self.random.integers(200, 321))

  def _collide_rocks(self) -> int:
    queued: dict[Rock, tuple[float, int]] = {}
    large_count = sum(rock.size > 4.5 for rock in self.rocks)
    for first, second in combinations(self.rocks, 2):
      dx, dy = second.x - first.x, second.y - first.y
      distance = math.hypot(dx, dy)
      if not 0.01 < distance < (first.size + second.size) / 2:
        continue
      source, target = (first, second) if first.hot and first.speed > 0.7 else (second, first)
      if source.hot and source.speed > 0.7 and target.size > 4.5 and target not in queued and large_count < 10:
        queued[target] = source.speed, max(1, source.depth) + 1
      else:
        vx, vy = dx * 0.3 / distance, dy * 0.3 / distance
        first.vx -= vx
        first.vy -= vy
        second.vx += vx
        second.vy += vy
    for target, (speed, depth) in queued.items():
      self._split(target, self.random.uniform(0, math.tau), 1.3, depth)
      self.freeze = max(self.freeze, 2 + (speed > 1.0) + (speed > 1.6))
    return sum(depth for _, depth in queued.values())

  def _split(self, rock: Rock, angle: float, speed: float, depth: int) -> None:
    self.rocks.remove(rock)
    for offset in (-0.6, 0.6):
      dx, dy = math.cos(angle + offset), math.sin(angle + offset)
      self.rocks.append(Rock(rock.x + 4 * dx, rock.y + 4 * dy,
                             speed * dx, speed * dy, rock.size * 0.62, 55, depth))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    player = self.player
    state[int(player.y) // 10, int(player.x) // 10, 0] = True
    x = player.x + 8 * math.cos(self.angle)
    y = player.y + 8 * math.sin(self.angle)
    state[min(9, max(0, int(y) // 10)), min(9, max(0, int(x) // 10)), 4] = True
    for rock in self.rocks:
      channel = 2 if rock.hot else 3 if rock.size <= 4.5 else 1
      state[min(9, int(rock.y) // 10), min(9, int(rock.x) // 10), channel] = True
    values = (self.steps, round(self.angle * 80), self.freeze,
              round((player.vx + 16) * 8), round((player.vy + 16) * 8))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
