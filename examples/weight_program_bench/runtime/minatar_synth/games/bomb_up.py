from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Body:
  x: float
  y: float
  vx: float = 0
  vy: float = 0


@dataclass
class Blast:
  x: float
  y: float
  age: int = 0

  @property
  def radius(self) -> float:
    return 25 * math.sin(self.age * 0.15)

  def hits(self, x: float, y: float, tolerance: float) -> bool:
    return abs(math.hypot(x - self.x, y - self.y) - self.radius) <= tolerance


class BombUpEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.player = Body(50, 50)
    self.spawner = Body(80, 0)
    self.bomb: Body | None = None
    self.blast: Blast | None = None
    self.rocks: list[Body] = []

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
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  @staticmethod
  def _bounce(body: Body) -> None:
    if (body.x < 7 and body.vx < 0) or (body.x > 93 and body.vx > 0):
      body.x = min(93, max(7, body.x))
      body.vx *= -0.7

  def _tap(self) -> None:
    player = self.player
    if self.bomb is None:
      direction = -1 if player.vx < 0 else 1
      self.bomb = Body(player.x + 2 * direction, player.y, player.vx + 0.5 * direction, player.vy)
      self.multiplier = 1
      return
    self.blast = Blast(self.bomb.x, self.bomb.y)
    dx, dy = player.x - self.bomb.x, player.y - self.bomb.y
    distance_squared = max(1, dx * dx + dy * dy)
    player.vx += 20 * dx / distance_squared
    player.vy += 20 * dy / distance_squared
    self.bomb = None

  def _advance_frame(self, tap: bool) -> int:
    player = self.player
    scroll = (40 - player.y) * 0.3 if player.y < 40 else min(0, (60 - player.y) * 0.2)
    self._bounce(player)
    player.vx *= 0.99
    player.vy = (player.vy + 0.02) * 0.99
    player.x += player.vx
    player.y += player.vy + scroll
    if tap:
      self._tap()
    if self.bomb is not None:
      bomb = self.bomb
      self._bounce(bomb)
      bomb.vx *= 0.98
      bomb.vy = (bomb.vy + 0.15) * 0.98
      bomb.x += bomb.vx
      bomb.y += bomb.vy + scroll
      if bomb.y > 103:
        self.bomb = None
    if self.blast is not None:
      self.blast.age += 1
      if self.blast.radius < 0:
        self.blast = None

    self.spawner.vx = (self.spawner.vx + self.random.uniform(-0.1, 0.1)) * 0.99
    self.spawner.x += self.spawner.vx
    self._bounce(self.spawner)
    if self.spawn_timer == 0:
      bottom = self.random.random() > (0.05 if player.y > 30 else 0.9)
      y = 103 if bottom else -3
      vy = -self.random.uniform(0.5, 1) if bottom else player.vy / 2
      self.rocks.append(Body(self.spawner.x, y, vy=vy))
      self.spawn_timer = int(self.random.integers(8, 11))
    self.spawn_timer -= 1

    reward = 0
    remaining = []
    for rock in self.rocks:
      rock.y += rock.vy
      rock.vy = (rock.vy + 0.01) * 0.99
      rock.y += rock.vy + scroll
      if self.blast is not None and self.blast.hits(rock.x, rock.y, 3.5):
        reward += self.multiplier
        self.multiplier += 1
        continue
      if abs(rock.x - player.x) < 5.5 and abs(rock.y - player.y) < 5.5:
        self.terminal = True
      if -50 <= rock.y <= 150:
        remaining.append(rock)
    self.rocks = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.player.y) // 10))
    column = min(9, max(0, int(self.player.x) // 10))
    state[row, column, 0] = True
    state[row, column, 4] = self.player.vy < 0
    if self.bomb is not None and 0 <= self.bomb.x < 100 and 0 <= self.bomb.y < 100:
      state[int(self.bomb.y) // 10, int(self.bomb.x) // 10, 1] = True
    for rock in self.rocks:
      if 0 <= rock.y < 100:
        state[int(rock.y) // 10, int(rock.x) // 10, 2] = True
    if self.blast is not None:
      for y in range(10):
        for x in range(10):
          state[y, x, 3] = self.blast.hits(x * 10 + 5, y * 10 + 5, 7)
    for row, value in enumerate((self.steps, self.multiplier)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 5] = self.pressed, self.bomb is not None, self.player.vx < 0
    return state
