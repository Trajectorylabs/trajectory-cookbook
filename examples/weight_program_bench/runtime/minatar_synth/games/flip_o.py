from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Ball:
  x: float = 80
  y: float = 10
  vx: float = 1
  vy: float = 0
  combo: int = 1


@dataclass
class Block:
  x: float
  y: float
  extra_ball: bool


@dataclass
class Bar:
  x: float
  y: float
  angle: float
  length: float

  def touches(self, x: float, y: float, radius: float) -> bool:
    cos, sin = math.cos(self.angle), math.sin(self.angle)
    dx, dy = x - self.x, y - self.y
    along = min(self.length, max(0, dx * cos + dy * sin))
    return math.hypot(dx - along * cos, dy - along * sin) < radius

  def bounce(self, ball: Ball, boost: bool = False) -> None:
    nx, ny = -math.sin(self.angle), math.cos(self.angle)
    if ny > 0:
      nx, ny = -nx, -ny
    inward = ball.vx * nx + ball.vy * ny
    if inward < 0:
      ball.vx -= 1.7 * inward * nx
      ball.vy -= 1.7 * inward * ny
    if boost:
      ball.vy -= 4
      ball.combo = 1
    for _ in range(8):
      if not self.touches(ball.x, ball.y, 3.5):
        break
      ball.x += nx
      ball.y += ny


RAMPS = (Bar(7, 75, 0.5, 25), Bar(94, 75, math.pi - 0.5, 25))


class FlipOEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.flipped = self.terminal = False
    self.next_block = 0.0
    self.balls = [Ball()]
    self.blocks: list[Block] = []

  def _flippers(self) -> tuple[Bar, Bar]:
    angle = -0.5 if self.flipped else 0.5
    return Bar(33, 88, angle, 12), Bar(68, 88, math.pi + angle, 12)

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

  def _advance(self, tap: bool) -> int:
    deepest = max((block.y for block in self.blocks), default=0)
    scroll = (30 - deepest) * 0.1 if deepest < 29 else 0.02
    if tap:
      scroll += 0.3 * len(self.balls)
      self.flipped = not self.flipped
    flippers = self._flippers()
    bars = RAMPS + flippers
    for block in self.blocks:
      block.y += scroll
      if any(bar.touches(block.x, block.y, 6) for bar in bars):
        self.terminal = True

    reward = 0
    count = len(self.balls)
    spawned = []
    for ball in self.balls:
      previous_y = ball.y + scroll
      ball.vy = (ball.vy + 0.1) * 0.99
      ball.vx *= 0.99
      ball.x += ball.vx * 0.5
      ball.y = previous_y + ball.vy * 0.5
      if ball.x < 7 or ball.x > 93:
        ball.x = min(93, max(7, ball.x))
        ball.vx *= -0.7
      if ball.y < 7:
        ball.y, ball.vy = 7, abs(ball.vy) * 0.7
      survivors = []
      for block in self.blocks:
        if abs(block.x - ball.x) < 6.5 and abs(block.y - ball.y) < 4.5:
          reward += ball.combo * count
          ball.combo += 1
          if abs(previous_y - block.y) >= 4.5:
            ball.vy *= -0.7
          else:
            ball.vx *= -0.7
          if block.extra_ball:
            spawned.append(Ball(block.x, block.y))
        else:
          survivors.append(block)
      self.blocks = survivors
      for bar in RAMPS:
        if bar.touches(ball.x, ball.y, 3.5):
          bar.bounce(ball)
      for bar in flippers:
        if bar.touches(ball.x, ball.y, 3.5):
          bar.bounce(ball, boost=tap)
    self.balls = [ball for ball in self.balls if ball.y <= 101] + spawned
    self.terminal |= not self.balls
    if self.balls:
      self._spawn(scroll)
    return reward

  def _spawn(self, scroll: float) -> None:
    self.next_block -= scroll
    while self.next_block < 0:
      for offset in (5, 15, 25, 35):
        if self.random.random() < 0.5:
          for x in (50 - offset, 50 + offset):
            self.blocks.append(Block(x, -self.next_block, self.random.random() < 0.1 / len(self.balls)))
      self.next_block += 6

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for ball in self.balls:
      if 0 <= ball.y < 100:
        state[int(ball.y) // 10, int(ball.x) // 10, 0] = True
    for block in self.blocks:
      if 0 <= block.y < 100:
        state[int(block.y) // 10, int(block.x) // 10, 2 if block.extra_ball else 1] = True
    for index, bar in enumerate(RAMPS + self._flippers()):
      for distance in np.linspace(0, bar.length, 10):
        x, y = bar.x + distance * math.cos(bar.angle), bar.y + distance * math.sin(bar.angle)
        state[int(y) // 10, int(x) // 10, 3 + int(index >= 2)] = True
    for row, value in enumerate((self.steps, len(self.balls))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.flipped, self.pressed
    return state
