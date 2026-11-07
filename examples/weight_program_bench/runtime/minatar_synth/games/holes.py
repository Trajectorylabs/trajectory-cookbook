from dataclasses import dataclass
import math

import numpy as np


SLOPE = math.tan(0.2)


@dataclass
class Ball:
  x: float
  y: float
  vx: float = 0.0
  vy: float = 0.0

  def reflect(self, nx: float, ny: float) -> None:
    approach = min(0, self.vx * nx + self.vy * ny)
    self.vx -= 1.7 * approach * nx
    self.vy -= 1.7 * approach * ny


@dataclass
class Ramp:
  x: float
  y: float
  direction: int
  length: float
  gate: int = -1

  def height_at(self, distance: float) -> float:
    return self.y + distance * SLOPE


class HolesEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.balls = [Ball(10, 0)]
    self.ramps: list[Ramp] = []
    self.gate = self.next_gate = self.steps = 0
    self.scroll, self.scroll_line = 0.0, 60.0
    self.pressed = self.terminal = False
    self._place_coin(20, 1)

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.gate ^= 1
      self.scroll_line -= 9
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _place_coin(self, y: float, direction: int) -> None:
    self.coin_x = 10 if direction == 1 else 89
    self.coin_y = y
    self.coin_ramp = Ramp(7 if direction == 1 else 93, y + 9, direction, 9)
    self.ramps.append(self.coin_ramp)

  def _extend(self) -> None:
    ramp = self.coin_ramp
    self.ramps.remove(ramp)
    holes = []
    for x in self.random.uniform(15, 64.5, 5):
      if all(abs(x - other) >= 21 for other in holes):
        holes.append(x)
    start = 0.0
    for x in sorted(holes):
      edge = x - 6
      self._segment(ramp, start, edge - start - 5)
      self._segment(ramp, edge, 7, self.next_gate)
      if self.random.random() < 0.7:
        self.next_gate ^= 1
      start = x + 6
    self._segment(ramp, start, 80 - start - 5)
    self._place_coin(ramp.y + 80 * math.sin(0.2) + 9, -ramp.direction)

  def _segment(self, ramp: Ramp, start: float, length: float, gate: int = -1) -> None:
    self.ramps.append(Ramp(ramp.x + start * ramp.direction, ramp.height_at(start),
                           ramp.direction, length, gate))

  def _advance(self) -> float:
    reward = self.scroll * len(self.balls)
    self.coin_y -= self.scroll
    for ramp in self.ramps:
      ramp.y -= self.scroll
      if ramp.y < 9:
        ramp.y -= 1
    self.ramps = [ramp for ramp in self.ramps if ramp is self.coin_ramp or ramp.height_at(ramp.length) >= 0]
    deepest = 0.0
    for ball in tuple(self.balls):
      ball.vy = (ball.vy + 0.2) * 0.98
      ball.vx *= 0.98
      ball.x += ball.vx * 0.5
      ball.y += ball.vy * 0.5 - self.scroll
      if ball.y < self.scroll_line + 20:
        deepest = max(deepest, ball.y)
      if math.hypot(ball.x - self.coin_x, ball.y - self.coin_y) < 5:
        reward += 10 * len(self.balls)
        if len(self.balls) < 9:
          self.balls.append(Ball(self.random.uniform(30, 70), 0, vy=1))
        self._extend()
      if ball.x < 6 or ball.x > 94:
        ball.reflect(1 if ball.x < 6 else -1, 0)
        ball.x = min(94, max(6, ball.x))
      for ramp in self.ramps:
        distance = (ball.x - ramp.x) * ramp.direction
        height = ramp.height_at(distance)
        if ramp.gate in (-1, self.gate) and -1 < distance < ramp.length + 1 and abs(ball.y - height) < 3:
          ball.reflect(math.sin(0.2) * ramp.direction, -math.cos(0.2))
          ball.y = height - 3
    self.balls = [ball for ball in self.balls if -1 <= ball.y <= 100]
    for ball in self.balls:
      for other in self.balls:
        dx, dy = ball.x - other.x, ball.y - other.y
        distance = math.hypot(dx, dy)
        if 0 < distance <= 2:
          ball.reflect(dx / distance, dy / distance)
    self.scroll = max(0, deepest - self.scroll_line) * 0.1
    self.scroll_line += (60 - self.scroll_line) * 0.01
    self.terminal = not self.balls
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for ball in self.balls:
      state[min(9, max(0, int(ball.y) // 10)), int(ball.x) // 10, 0] = True
    for ramp in self.ramps:
      channel = 1 if ramp.gate < 0 else 2 + ramp.gate
      for distance in range(0, math.ceil(ramp.length) + 1, 2):
        x, y = ramp.x + distance * ramp.direction, ramp.height_at(distance)
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, channel] = True
    if 0 <= self.coin_y < 100:
      state[int(self.coin_y) // 10, self.coin_x // 10, 4] = True
    for row, value in enumerate((self.steps, len(self.balls), round(self.scroll_line) + 256)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, bool(self.gate)
    return state
