from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Spark:
  x: float
  y: float
  vx: float
  vy: float
  blue: bool
  size: float = 1.0


class HoardSpoutEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 50.0
    self.angle = self.random.uniform(0, math.tau)
    self.speed, self.size = 1.0, 3.0
    self.spawn_timer = 0.0
    self.steps = 0
    self.blue = self.pressed = self.terminal = False
    self.sparks: list[Spark] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, released: bool) -> int:
    self.x += math.cos(self.angle) * self.speed
    self.y += math.sin(self.angle) * self.speed
    if released:
      self.blue = not self.blue
    if self.pressed:
      if self.speed < 2:
        self.speed += 0.01
      self.angle += 0.01
    else:
      self.speed += (1 - self.speed) * 0.1
    self.spawn_timer -= 1.1 if self.pressed else 0.1
    if self.spawn_timer < 0:
      angle = self.angle + math.pi + self.random.uniform(-0.2, 0.2)
      self.sparks.append(Spark(self.x, self.y, math.cos(angle), math.sin(angle), not self.blue))
      self.spawn_timer = 9
    if not 0 <= self.x <= 100:
      self.x = min(100, max(0, self.x))
      self.angle = math.pi - self.angle
    if not 0 <= self.y <= 100:
      self.y = min(100, max(0, self.y))
      self.angle = -self.angle
    self.size = min(9, self.size - 0.002)
    if self.size < 1:
      self.terminal = True
      return 0
    reward = 0
    remaining = []
    for spark in self.sparks:
      spark.x += spark.vx
      spark.y += spark.vy
      spark.size += 0.2 if spark.size < 5 else 0.01
      if spark.size > 9:
        continue
      if spark.size >= 5 and self._hits_player(spark):
        if spark.blue == self.blue:
          reward += math.floor(self.size)
          self.size += 0.2
        else:
          self.size -= 0.4
        continue
      if not 0 <= spark.x <= 100:
        spark.x = min(100, max(0, spark.x))
        spark.vx *= -1
      if not 0 <= spark.y <= 100:
        spark.y = min(100, max(0, spark.y))
        spark.vy *= -1
      remaining.append(spark)
    self.sparks = remaining
    return reward

  def _hits_player(self, spark: Spark) -> bool:
    dx, dy = spark.x - self.x, spark.y - self.y
    cosine, sine = math.cos(self.angle), math.sin(self.angle)
    return (abs(dx * cosine + dy * sine) < (self.size + 3 + spark.size) / 2
            and abs(-dx * sine + dy * cosine) < (self.size + spark.size) / 2)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y) // 10), min(9, int(self.x) // 10), 0] = True
    for spark in self.sparks:
      channel = 1 + 2 * spark.blue + (spark.size >= 5)
      state[min(9, int(spark.y) // 10), min(9, int(spark.x) // 10), channel] = True
    for row, value in enumerate((self.steps, round(self.size * 50),
                                 round((self.angle % math.tau) * 80), round(self.speed * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.blue
    return state
