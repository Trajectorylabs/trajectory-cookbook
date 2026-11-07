from dataclasses import dataclass
import math

import numpy as np


Y = 60


@dataclass(eq=False)
class Enemy:
  x: float
  y: float
  angle: float
  speed: float
  beam: float = 0

  def tip(self) -> tuple[float, float]:
    return self.x + math.cos(self.angle) * self.beam, self.y + math.sin(self.angle) * self.beam


@dataclass
class Counter:
  x: float
  radius: float
  enemy: Enemy | None


class CounterBEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.shield = 0
    self.spawn_timer = 9
    self.multiplier = 1
    self.x, self.base_x = 20.0, 30.0
    self.speed = self.beam = 0.0
    self.pressed = self.terminal = False
    self.counter: Counter | None = None
    self.enemies = [Enemy(205, Y, math.pi, 0.5)]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    scroll = (self.x - self.base_x) * 0.1
    self.base_x += (30 - self.base_x) * 0.001
    self.speed += ((0.2 if self.pressed else 2) - self.speed) * 0.2
    self.x += self.speed - scroll
    if self.counter is None:
      self.beam = min(300, self.beam + 4) if self.pressed else 0
    else:
      self.counter.x -= scroll
      self.beam = max(0, self.counter.x - self.x)
      if not self.beam and self.counter.enemy is not None:
        self.counter.enemy.beam = -9999
        self.counter.enemy = None
      growth = 1 if self.pressed and self.counter.enemy is not None else -2
      self.counter.radius = min(30, self.counter.radius + growth)
      if self.counter.radius < 1:
        if self.counter.enemy is not None:
          self.counter.enemy.beam = -9999
        self.counter = None

    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self._spawn()
    reward = 0
    incoming = False
    survivors = []
    for enemy in self.enemies:
      enemy.x += math.cos(enemy.angle) * enemy.speed - scroll
      enemy.y += math.sin(enemy.angle) * enemy.speed
      if self.counter is not None and self.counter.enemy is enemy:
        enemy.beam = math.hypot(enemy.x - self.counter.x, enemy.y - Y)
        enemy.speed *= 0.8
      else:
        enemy.beam += 0.8
      if enemy.beam > 0:
        tx, ty = enemy.tip()
        if self.counter is None and self.beam and abs(tx - self.x - self.beam) < 5 and abs(ty - Y) < 5:
          self.counter = Counter(self.x + self.beam, 1, enemy)
          self.multiplier = 1
        if (self.counter is None or self.counter.enemy is not enemy) and (
          ty > 90 or tx < -99 or self._counter_hits(tx, ty)
        ):
          enemy.beam = -9999
      if self._counter_hits(enemy.x, enemy.y):
        reward += self.multiplier
        self.multiplier = min(64, self.multiplier * 2)
        if self.counter is not None and self.counter.enemy is enemy:
          self.counter.enemy = None
      elif self.beam and self.x <= enemy.x <= self.x + self.beam and abs(enemy.y - Y) < 3:
        if self.counter is not None and self.counter.enemy is enemy:
          self.counter.enemy = None
      elif enemy.x >= -5:
        incoming |= self._beam_hits_player(enemy)
        if enemy.beam < -999:
          enemy.speed += (1 - enemy.speed) * 0.1
          enemy.angle += (math.pi - enemy.angle) * 0.05
        survivors.append(enemy)
    self.enemies = survivors
    self.shield = max(0, self.shield - 1)
    if incoming and not self.shield and (
      self.counter is None or abs(self.x - self.counter.x) >= self.counter.radius + 9
    ):
      self.x -= 10
      self.base_x -= 10
      self.shield = 60
    self.terminal = self.x < 0
    return reward

  def _counter_hits(self, x: float, y: float) -> bool:
    return self.counter is not None and abs(
      math.hypot(x - self.counter.x, y - Y) - self.counter.radius
    ) < 4

  def _beam_hits_player(self, enemy: Enemy) -> bool:
    if enemy.beam <= 0:
      return False
    dx, dy = self.x - enemy.x, Y - enemy.y
    cos, sin = math.cos(enemy.angle), math.sin(enemy.angle)
    distance = min(enemy.beam, max(0, dx * cos + dy * sin))
    return math.hypot(dx - distance * cos, dy - distance * sin) < 3

  def _spawn(self) -> None:
    if self.random.random() < 0.6:
      x, y = self.random.uniform(150, 300), -5
    else:
      x, y = 205, self.random.uniform(0, 85)
    target = self.x + 9 if self.random.random() < 0.05 else x - self.random.uniform(30, 150)
    self.enemies.append(Enemy(x, y, math.atan2(Y - y, target - x) % math.tau,
                              1, -self.random.uniform(0, 20)))
    self.spawn_timer = int(self.random.integers(30, 41))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    column = min(9, max(0, int(self.x) // 20))
    state[6, column, 0] = True
    if self.beam:
      state[6, column:min(10, int(self.x + self.beam) // 20 + 1), 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200 and 0 <= enemy.y < 100:
        state[int(enemy.y) // 10, int(enemy.x) // 20, 2] = True
      if enemy.beam > 0:
        tx, ty = enemy.tip()
        for fraction in np.linspace(0, 1, 16):
          x = enemy.x + (tx - enemy.x) * fraction
          y = enemy.y + (ty - enemy.y) * fraction
          if 0 <= x < 200 and 0 <= y < 100:
            state[int(y) // 10, int(x) // 20, 3] = True
    for row in range(10):
      for column in range(10):
        state[row, column, 4] = self._counter_hits(column * 20 + 10, row * 10 + 5)
    for row, value in enumerate((self.steps, self.multiplier, round(self.beam), self.shield)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
