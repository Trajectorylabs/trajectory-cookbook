from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Block:
  x: float
  y: float
  width: float


class BlockStackerEnvironment:
  score_cap = 50

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.stacked = 0
    self.pressed = self.falling = self.terminal = False
    self.angle = self.vx = self.vy = self.drift = 0.0
    self.current = Block(50, 25, self.random.uniform(10, 30))
    self.tower = [Block(50, 99, 99)]

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

  def _advance_frame(self, tap: bool) -> int:
    difficulty = 1 + self.stacked / 50
    self.angle += 0.05 * difficulty
    previous_y = self.current.y
    if self.falling:
      self.current.x += self.vx * difficulty
      self.current.y += self.vy * difficulty
      self.vy += 0.1
    else:
      swing = math.sin(self.angle)
      self.current.x = 50 - 20 * math.sin(swing)
      self.current.y = 5 + 20 * math.cos(swing)
      if tap:
        self.falling = True
        self.vx, self.vy = -math.cos(self.angle), 0.0

    for block in self.tower:
      block.x = (block.x + self.drift) % 100
    top = min(block.y for block in self.tower)
    reward = 0
    if self.falling:
      for block in sorted(self.tower, key=lambda block: block.y):
        distance = abs((self.current.x - block.x + 50) % 100 - 50)
        if previous_y <= block.y - 10 <= self.current.y and distance < (self.current.width + block.width) / 2:
          self.current.y = block.y - 10
          self.tower.append(self.current)
          self.current = Block(50, 25, self.random.uniform(10, 30))
          self.falling = False
          self.stacked += 1
          self.drift = self.random.uniform(-1, 1) * self.stacked / 50
          reward = 1
          break
      else:
        self.terminal = self.current.y > 105
    scroll = difficulty * 0.01 + max(0, 70 - top) * 0.1
    for block in self.tower:
      block.y += scroll
    self.tower = [block for block in self.tower if block.y <= 106]
    self.terminal |= not self.tower
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for block in self.tower:
      self._draw_block(state, block, 0, wraps=True)
    self._draw_block(state, self.current, 2 if self.falling else 1, wraps=False)
    if not self.falling:
      for fraction in np.linspace(0, 1, 8):
        x = 50 + (self.current.x - 50) * fraction
        y = (self.current.y - 5) * fraction
        state[int(y) // 10, int(x) // 10, 3] = True
    if self.drift < 0:
      state[:, :, 4] = state[:, :, 0]
    state[9, 0, 5] = self.pressed
    for row, value in enumerate((self.steps, self.stacked)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 1, 5] = (self.vx < 0 if self.falling else math.cos(self.angle) > 0)
    return state

  @staticmethod
  def _draw_block(state: np.ndarray, block: Block, channel: int, wraps: bool) -> None:
    for row in range(10):
      if row * 10 < block.y + 5 and (row + 1) * 10 > block.y - 5:
        for column in range(10):
          dx = column * 10 + 5 - block.x
          distance = abs((dx + 50) % 100 - 50) if wraps else abs(dx)
          state[row, column, channel] |= distance < block.width / 2 + 5
