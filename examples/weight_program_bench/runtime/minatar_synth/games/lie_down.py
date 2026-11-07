from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Emitter:
  x: float
  speed: float
  height: float = 0


@dataclass
class Rock:
  x: float
  target_speed: float
  speed: float = 0


class LieDownEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.speed = 20.0, 0.0
    self.fall = self.steps = self.hole_timer = 0
    self.enemy_timer = 60
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.holes: list[Emitter] = []
    self.enemies: list[Emitter] = []
    self.rocks: list[Rock] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed and not self.fall:
      self.multiplier = max(1, self.multiplier - 1)
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    scroll = 0 if self.fall else 0.1 + max(0, self.x - 30) * 0.1
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      speed = 0.1 + self.random.uniform(0, 2) * self.random.uniform(0, 2) * 0.1
      self.enemies.append(Emitter(self.random.uniform(100, 250), speed))
      self.enemy_timer = 50
    for rock in self.rocks:
      rock.x -= scroll + rock.speed
      rock.speed += (rock.target_speed - rock.speed) * 0.1
    self.rocks = [rock for rock in self.rocks if -3 <= rock.x <= 203]
    self.hole_timer -= 1
    if self.hole_timer < 0:
      self.holes.append(Emitter(self.random.uniform(0, 300), self.random.uniform(0.1, 0.2)))
      self.hole_timer = 100
    for hole in self.holes:
      hole.x -= scroll
      hole.height += hole.speed
      if hole.height > 11 and hole.speed > 0:
        hole.speed *= -1
    self.holes = [hole for hole in self.holes if hole.height >= 0]
    if self.fall:
      self.fall += 1
      self.terminal = self.fall > 8
    else:
      self.speed = (self.speed + (0 if self.pressed else 0.02)) * 0.99
      self.x += self.speed - scroll
      self.terminal = self.x < 0 or (not self.pressed and any(
        abs(rock.x - self.x) < 4 and rock.speed > rock.target_speed * 0.5 for rock in self.rocks
      ))
      if any(hole.height > 9 and abs(hole.x + 4.5 - self.x) < 2 for hole in self.holes):
        self.speed = 0
        if self.pressed:
          self.x -= 5
        else:
          self.fall = 1
    reward = 0
    remaining = []
    for enemy in self.enemies:
      enemy.x -= scroll
      enemy.height += math.sqrt(enemy.speed * 10) / 10 if enemy.speed > 0 else enemy.speed
      if enemy.height > 15 and enemy.speed > 0:
        self.rocks.append(Rock(enemy.x, enemy.speed * 3))
        enemy.speed *= -1
      y = 37 + (10 - min(10, max(0, enemy.height))) ** 2 / 10
      player_y = 37 + self.fall if self.fall else (39 if self.pressed else 37)
      if abs(enemy.x - self.x) < 5 and abs(y - player_y) < (4 if self.pressed else 6):
        reward += self.multiplier
        self.multiplier += 1
      elif enemy.height >= 0:
        remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, (37 + self.fall + 2 * self.pressed) // 5),
          min(9, max(0, int(self.x // 20))), 0] = True
    for channel, objects in ((1, self.holes), (2, self.enemies), (3, self.rocks)):
      for obj in objects:
        if 0 <= obj.x < 200:
          state[(8, 7, 6)[channel - 1], int(obj.x // 20), channel] = True
    state[8:, :, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.fall, round(self.speed * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
