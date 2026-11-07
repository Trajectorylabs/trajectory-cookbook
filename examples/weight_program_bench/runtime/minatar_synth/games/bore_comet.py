from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
PLAYER_X = 35
SCROLL = 0.6


@dataclass
class Rock:
  x: float
  y: float
  width: float
  height: float
  bore: int = 0

  @property
  def threshold(self) -> int:
    return round(math.sqrt(self.width * self.height) / (SCROLL / 0.5))

  def contains(self, y: float) -> bool:
    return abs(self.x - PLAYER_X) < self.width / 2 + 2 and abs(self.y - y) < self.height / 2 + 2


@dataclass
class Enemy:
  x: float
  y: float
  vx: float


class BoreCometEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.freeze = 0
    self.pressed = self.terminal = self.was_in_rock = False
    self.y, self.vy = 30.0, 0.0
    self.next_rock, self.next_enemy = 20.0, 60
    self.rocks: list[Rock] = []
    self.enemies: list[Enemy] = []

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
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    if self.freeze:
      self.freeze -= 1
      return 0
    rock = next((rock for rock in reversed(self.rocks) if rock.contains(self.y)), None)
    if rock is not None and not self.pressed:
      self.terminal = True
      return 0

    reward = 0
    if self.pressed:
      self.vy -= 0.06 if rock is not None else 0.11
      if rock is not None:
        rock.bore += 1
        reward += 1
        if rock.bore >= rock.threshold:
          reward += rock.threshold + self._blast(15 + (rock.width + rock.height) / 4, 3)
          self.rocks.remove(rock)
          rock = None
    self.vy = min(1.8, max(-1.5, self.vy + 0.05))
    self.y += self.vy * (0.45 if rock is not None else 1)
    if self.y < 10:
      self.y, self.vy = 10, 0
    if self.was_in_rock and rock is None and self.pressed:
      reward += self._blast(22, 2)
    self.was_in_rock = rock is not None

    self._spawn()
    for obstacle in self.rocks:
      obstacle.x -= SCROLL
    self.rocks = [obstacle for obstacle in self.rocks if obstacle.x >= -20]
    for enemy in self.enemies:
      enemy.x += enemy.vx
      if rock is None and math.hypot(enemy.x - PLAYER_X, enemy.y - self.y) < 5:
        self.terminal = True
    self.enemies = [enemy for enemy in self.enemies if enemy.x >= -8]
    self.terminal |= self.y > 91
    return reward

  def _blast(self, radius: float, freeze: int) -> int:
    survivors = [enemy for enemy in self.enemies if math.hypot(enemy.x - PLAYER_X, enemy.y - self.y) >= radius]
    kills = len(self.enemies) - len(survivors)
    self.enemies = survivors
    if kills or freeze == 3:
      self.freeze = max(self.freeze, min(8, freeze + kills))
    return 3 * kills**2

  def _spawn(self) -> None:
    self.next_rock -= SCROLL
    if self.next_rock < 0:
      self.rocks.append(Rock(112, self.random.uniform(25, 75), self.random.uniform(14, 30), self.random.uniform(10, 22)))
      self.next_rock = self.random.uniform(30, 55)
    self.next_enemy -= 1
    if self.next_enemy < 0:
      self.enemies.append(Enemy(105, self.random.uniform(8, 84), -SCROLL - self.random.uniform(0.2, 0.5)))
      self.next_enemy = int(self.random.integers(50, 91))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(self.y) // 10
    state[row, 3, 0] = True
    state[row, 3, 4] = self.pressed
    state[9, :, 3] = True
    for rock in self.rocks:
      for row in range(10):
        if abs(row * 10 + 5 - rock.y) < rock.height / 2 + 5:
          for column in range(10):
            state[row, column, 1] |= abs(column * 10 + 5 - rock.x) < rock.width / 2 + 5
    for enemy in self.enemies:
      if 0 <= enemy.x < 100:
        state[int(enemy.y) // 10, int(enemy.x) // 10, 2] = True
    active_rock = next((rock for rock in reversed(self.rocks) if rock.contains(self.y)), None)
    for row, value in enumerate((self.steps, self.freeze, active_rock.bore if active_rock else 0)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.vy < 0, self.was_in_rock
    return state
