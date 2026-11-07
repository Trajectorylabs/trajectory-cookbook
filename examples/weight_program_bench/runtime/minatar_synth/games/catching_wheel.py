from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
SPOKES = 6


@dataclass
class Falling:
  x: float
  y: float = 0
  vy: float = 0

  def advance(self) -> None:
    self.vy += 0.02
    self.y += self.vy

  def near(self, x: float, y: float, radius: float) -> bool:
    return math.hypot(self.x - x, self.y - y) < radius


class CatchingWheelEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.angle = 0.0
    self.human_timer, self.obstacle_timer = 40.0, 99.0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.baskets = [True] * SPOKES
    self.humans: list[Falling] = []
    self.obstacles: list[Falling] = []

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

  def _positions(self) -> list[tuple[float, float]]:
    angles = (self.angle + index * math.tau / SPOKES for index in range(SPOKES))
    return [(50 + 40 * math.cos(angle), 90 + 40 * math.sin(angle)) for angle in angles]

  def _change_basket(self, restore: bool) -> None:
    start = int(self.random.integers(SPOKES))
    for offset in range(SPOKES):
      index = (start + offset) % SPOKES
      if self.baskets[index] != restore:
        self.baskets[index] = restore
        if not restore:
          self.multiplier = 1
        break

  def _advance_frame(self) -> int:
    self.obstacle_timer -= 1
    if self.obstacle_timer <= 0:
      self.obstacles.append(Falling(self.random.uniform(20, 80)))
      self.obstacle_timer = self.random.uniform(79.2, 118.8)
    for obstacle in self.obstacles:
      obstacle.advance()
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y <= 95]

    if self.pressed:
      self.angle = (self.angle + 0.05) % math.tau
    positions = self._positions()
    for index, (x, y) in enumerate(positions):
      if self.baskets[index] and any(obstacle.near(x, y, 9) for obstacle in self.obstacles):
        self.baskets[index] = False
        self.multiplier = 1
    catchable = [point for point, live in zip(positions, self.baskets) if live]

    self.human_timer -= 1
    if self.human_timer <= 0:
      self.humans.append(Falling(self.random.uniform(10, 90)))
      self.human_timer = self.random.uniform(28, 40)
    reward = 0
    remaining = []
    for human in self.humans:
      human.advance()
      if any(obstacle.near(human.x, human.y, 5) for obstacle in self.obstacles):
        continue
      if any(human.near(x, y, 6) for x, y in catchable):
        reward += self.multiplier
        self.multiplier += 1
        self._change_basket(restore=True)
      elif human.y > 100:
        self._change_basket(restore=False)
      else:
        remaining.append(human)
    self.humans = remaining
    self.terminal = not any(self.baskets)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 5, 0] = True
    for live, (x, y) in zip(self.baskets, self._positions()):
      if 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, 1 if live else 2] = True
    for channel, objects in ((3, self.humans), (4, self.obstacles)):
      for obj in objects:
        if obj.y < 100:
          state[int(obj.y) // 10, int(obj.x) // 10, channel] = True
    for row, value in enumerate((self.steps, self.multiplier, round(self.angle * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :SPOKES, 5] = self.baskets
    state[9, SPOKES, 5] = self.pressed
    return state
