from dataclasses import dataclass

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
GROUND = 86


@dataclass
class Barrel:
  x: float
  y: float = -5
  velocity: float = 0
  rolling: bool = False


class BarollEnvironment:
  score_cap = HORIZON * FRAMES_PER_STEP

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.pressed = self.terminal = self.jumping = False
    self.y = float(GROUND)
    self.velocity = 0.0
    self.barrels: list[Barrel] = []

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
      self._advance_frame(tap and frame == 0)
      reward += int(not pressed)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, tap: bool) -> None:
    if self.spawn_timer == 0:
      start = 100 if self.jumping else 10
      self.barrels.append(Barrel(self.random.uniform(start, 200)))
      self.spawn_timer = int(self.random.integers(30, 90))
    self.spawn_timer -= 1
    for barrel in self.barrels:
      if barrel.rolling:
        barrel.x -= 1
      else:
        barrel.velocity = (barrel.velocity + 0.2) * 0.92
        barrel.y += barrel.velocity
        if barrel.y > 85:
          barrel.y = GROUND
          barrel.rolling = True
      barrel.x -= 1 if self.pressed else 2
    self.barrels = [barrel for barrel in self.barrels if barrel.x > -5]

    if self.jumping:
      self.y += self.velocity
      self.velocity += 0.1 if self.pressed else 0.2
      if self.y > 85:
        self.y = GROUND
        self.jumping = self.pressed
        self.velocity = -3 if self.pressed else 0
    elif tap:
      self.jumping = True
      self.velocity = -3.6
    self.terminal = any(
      abs(barrel.x - 9) < 5 and abs(barrel.y - self.y) < 6
      for barrel in self.barrels
    )

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(self.y) // 10
    state[row, 0, 0] = True
    state[row, 0, 4] = self.jumping and self.velocity < 0
    for barrel in self.barrels:
      if barrel.x >= 0 and barrel.y >= 0:
        state[int(barrel.y) // 10, int(barrel.x) // 20, 2 if barrel.rolling else 1] = True
    state[9, :, 3] = True
    state[9, 0, 5] = self.pressed
    for bit in range(9):
      state[9, bit + 1, 5] = bool(self.steps & (1 << bit))
    return state
