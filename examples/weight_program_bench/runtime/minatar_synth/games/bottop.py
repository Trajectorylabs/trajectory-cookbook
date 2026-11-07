from dataclasses import dataclass

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Spike:
  x: float
  y: float


class BottopEnvironment:
  score_cap = HORIZON * FRAMES_PER_STEP / 10

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.spawn_distance = self.height = self.velocity = 0.0
    self.jumping = self.pressed = self.terminal = False
    self.spikes: list[Spike] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    frames = 0
    for _ in range(FRAMES_PER_STEP):
      self._advance_frame()
      frames += 1
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return frames / 10, self.terminal

  def _advance_frame(self) -> None:
    self.spawn_distance -= 1
    if self.spawn_distance < 0:
      if self.random.random() < 0.33:
        y = float(self.random.choice((8, 92)))
      else:
        y = self.random.uniform(8, 92)
      self.spikes.append(Spike(103, y))
      self.spawn_distance += self.random.uniform(30, 60)
    for spike in self.spikes:
      spike.x -= 1
    self.spikes = [spike for spike in self.spikes if spike.x > -3]

    if not self.jumping and self.pressed:
      self.jumping = True
      self.velocity = 3
    if self.jumping:
      self.velocity -= 0.1 if self.pressed else 0.3
      self.height += self.velocity
      if self.height < 0:
        self.height = self.velocity = 0
        self.jumping = False
    top, bottom = 8 + self.height, 92 - self.height
    self.terminal = any(
      abs(spike.x - 9) < 6 and min(abs(spike.y - top), abs(spike.y - bottom)) < 6
      for spike in self.spikes
    )

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(8 + self.height) // 10, 0, 0] = True
    state[int(92 - self.height) // 10, 0, 1] = True
    for spike in self.spikes:
      if 0 <= spike.x < 100:
        state[int(spike.y) // 10, int(spike.x) // 10, 2] = True
    state[(0, 9), :, 3] = True
    state[0, :3, 4] = self.pressed, self.jumping, self.velocity > 0
    for bit in range(9):
      state[9, bit, 5] = bool(self.steps & (1 << bit))
    return state
