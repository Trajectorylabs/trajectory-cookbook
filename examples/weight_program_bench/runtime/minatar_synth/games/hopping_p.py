from dataclasses import dataclass
import math

import numpy as np


PATTERNS = ((1, 1, 0.65), (1, 0.8, 2), (1.6, 0.6, 2), (1, 2, 1.8))


@dataclass
class Hopper:
  x: float
  y: float
  direction: int
  hop_x: float
  hop_y: float
  gravity: float
  vx: float = 0
  vy: float = 0
  rest: int = 0
  power: bool = False

  def turn(self) -> None:
    self.direction *= -1
    self.vx *= -1

  def advance(self, fast: bool = False) -> float:
    if self.rest:
      self.rest -= 1
      if self.rest:
        return self.y - 10 * abs(self.rest - 5) / 5
      self.vx = self.hop_x * self.direction
      self.vy = self.hop_y
    for _ in range(2 if fast else 1):
      self.x += self.vx
      self.y += self.vy
      self.vy += self.gravity
    if self.y > 90:
      self.y = 90
      self.rest = 9
    if (self.x < 0 and self.direction < 0) or (self.x > 199 and self.direction > 0):
      self.turn()
    return self.y - 10


class HoppingPEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = Hopper(99, 60, 1, 0.5, -1.2, 0.02, vx=0.5)
    self.player_y = 50.0
    self.hoppers: list[Hopper] = []
    self.enemy_timer, self.pickup_timer = 0.0, 200.0
    self.power_ticks = self.steps = 0
    self.multiplier = 1
    self.first_pickup = True
    self.pressed = self.terminal = False

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

  def _spawn(self, power: bool, weak: bool = False) -> None:
    direction = 1 if self.random.random() < 0.5 else -1
    pattern = (0.88, 1.25, 1.25) if power and self.first_pickup else PATTERNS[self.random.integers(4)]
    sx, sy, sg = pattern
    scale = 0.8 if power else 1
    self.hoppers.append(Hopper(-5 if direction == 1 else 205, 89, direction,
                               0.5 * sx * scale * (0.5 if weak else 1),
                               -1.2 * sy * scale, 0.02 * sg * scale, power=power))

  def _advance(self, tap: bool) -> int:
    weak = self.power_ticks > 60
    self.enemy_timer -= 3 if weak else 1
    if self.enemy_timer < 0:
      self._spawn(False, weak)
      self.enemy_timer = self.random.uniform(100, 150) * math.sqrt(1 + len(self.hoppers))
    self.pickup_timer -= 1
    if self.pickup_timer < 0:
      self._spawn(True)
      self.first_pickup = False
      self.pickup_timer = self.random.uniform(700, 999)
    self.power_ticks = max(0, self.power_ticks - 1)
    if not self.power_ticks:
      self.multiplier = 1
    if tap:
      self.player.turn()
    self.player_y = self.player.advance(self.power_ticks > 0) - (2 if self.power_ticks else 0)
    reward = 0
    remaining = []
    for hopper in self.hoppers:
      y = hopper.advance() - (2 if hopper.power else 0)
      extent = 3 * (1 + bool(self.power_ticks) + hopper.power)
      if abs(self.player.x - hopper.x) >= extent or abs(self.player_y - y) >= extent:
        remaining.append(hopper)
      elif hopper.power:
        self.power_ticks = int(300 * (1 + len(self.hoppers)) ** 0.25)
        for other in self.hoppers:
          other.turn()
      elif self.power_ticks:
        reward += self.multiplier
        self.multiplier = min(64, self.multiplier * 2)
      else:
        self.terminal = True
        remaining.append(hopper)
    self.hoppers = remaining
    if not self.hoppers:
      self.enemy_timer = 0
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.player_y) // 10)), min(9, max(0, int(self.player.x) // 20)), 0] = True
    for hopper in self.hoppers:
      if 0 <= hopper.x < 200:
        height = hopper.y - (10 * abs(hopper.rest - 5) / 5 if hopper.rest else 10)
        row, col = min(9, max(0, int(height) // 10)), int(hopper.x) // 20
        state[row, col, 3 if hopper.power else 1 + (hopper.direction < 0)] = True
    state[9, :, 4] = True
    values = (self.steps, min(511, self.power_ticks), self.multiplier, self.player.rest)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.player.direction > 0
    return state
