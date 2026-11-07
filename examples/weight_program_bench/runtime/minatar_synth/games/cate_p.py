from dataclasses import dataclass

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
DIRECTIONS = ((1, 0), (0.7, 0.7), (0, 1), (-0.7, 0.7),
              (-1, 0), (-0.7, -0.7), (0, -1), (0.7, -0.7))


@dataclass
class Mover:
  x: float
  y: float
  direction: int

  def move(self, speed: float) -> None:
    dx, dy = DIRECTIONS[self.direction]
    self.x += speed * dx
    self.y += speed * dy

  def inside(self, low: float, high: float) -> bool:
    return low <= self.x <= high and low <= self.y <= high

  def clamp(self) -> None:
    self.x = min(90, max(9, self.x))
    self.y = min(90, max(9, self.y))

  def touches(self, other: "Mover") -> bool:
    return abs(self.x - other.x) < 5 and abs(self.y - other.y) < 5


@dataclass
class Cate(Mover):
  entering: bool = True


class CatePEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.wave_count = self.wave_timer = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.player = Mover(50, 50, 0)
    self.shot: Mover | None = None
    self.entrance: Mover | None = None
    self.newest: Cate | None = None
    self.cates: list[Cate] = []

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
    player = self.player
    if tap or not player.inside(9, 90):
      if tap and self.shot is None:
        self.shot = Mover(player.x, player.y, player.direction)
        self.multiplier = 1
      player.clamp()
      player.direction = (player.direction + 2) % 8
    player.move(0.2)
    if self.shot is not None:
      if self.shot.inside(9, 91):
        self.shot.move(1.5)
      else:
        self.shot = None

    self._spawn()
    survivors = []
    reward = 0
    for cate in self.cates:
      cate.move(0.2)
      if cate.entering:
        cate.entering = not cate.inside(9, 90)
      elif not cate.inside(9, 90):
        corner = not 9 <= cate.x <= 90 and not 9 <= cate.y <= 90
        cate.direction = (cate.direction + (4 if corner else 3)) % 8
        cate.clamp()
      if self.shot is not None and cate.touches(self.shot):
        reward += self.multiplier
        self.multiplier += 1
        if self.newest is cate:
          self.newest = None
      else:
        self.terminal |= cate.touches(player)
        survivors.append(cate)
    self.cates = survivors
    return reward

  def _spawn(self) -> None:
    if self.wave_timer <= 0:
      direction = int(self.random.integers(4)) * 2
      offset = self.random.uniform(9, 90)
      x, y = ((-3, offset), (offset, -3), (103, offset), (offset, 103))[direction // 2]
      self.entrance = Mover(x, y, direction)
      self.wave_count = int(self.random.integers(5, 9))
      self.wave_timer = int(self.random.integers(60, 91))
    if self.wave_count == 0:
      self.wave_timer -= 1
    if self.newest is not None and self.newest.inside(3, 97):
      self.newest = None
    if self.wave_count and self.newest is None and self.entrance is not None:
      self.newest = Cate(self.entrance.x, self.entrance.y, self.entrance.direction)
      self.cates.append(self.newest)
      self.wave_count -= 1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, movers in ((0, [self.player]), (1, self.cates), (2, [self.shot])):
      for mover in movers:
        if mover is not None and mover.inside(0, 99.999):
          row, col = int(mover.y) // 10, int(mover.x) // 10
          state[row, col, channel] = True
          if isinstance(mover, Cate):
            state[row, col, 3] |= mover.entering
    state[(0, 9), :, 4] = True
    state[:, (0, 9), 4] = True
    for row, value in enumerate((self.steps, self.player.direction, self.multiplier,
                                 self.shot.direction if self.shot is not None else 0)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
