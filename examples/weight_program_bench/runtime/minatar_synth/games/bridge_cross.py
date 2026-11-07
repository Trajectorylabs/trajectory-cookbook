from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
BRIGADE_LEFT = 10
BRIGADE_RIGHT = 40
BRIDGE_TOP = 32
GROUND = 37


@dataclass
class Bridge:
  x: float
  length: float

  def overlaps_brigade(self) -> bool:
    return self.x <= BRIGADE_RIGHT and self.x + self.length >= BRIGADE_LEFT


class BridgeCrossEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.score_timer = 0
    self.spawn_timer = 0.0
    self.multiplier = 1
    self.y, self.vy = float(GROUND), 0.0
    self.pressed = self.jumping = self.falling = self.on_bridge = self.terminal = False
    self.bridges: list[Bridge] = []

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
    if self.spawn_timer <= 0:
      self.bridges.append(Bridge(153, self.random.uniform(20, 40)))
      self.spawn_timer = self.random.uniform(50, 100)
    self.spawn_timer -= 1
    for bridge in self.bridges:
      bridge.x -= 1
    self.bridges = [bridge for bridge in self.bridges if bridge.x + bridge.length > 0]
    above_bridge = any(bridge.overlaps_brigade() for bridge in self.bridges)

    if self.y < 35 and not above_bridge:
      self.falling, self.on_bridge = True, False
    if self.jumping or self.falling:
      self.y += self.vy
      self.vy += 0.1 if self.pressed else 0.2
      surface = BRIDGE_TOP if above_bridge else GROUND
      if self.y > surface and self.vy > 0:
        self.y, self.vy = float(surface), 0.0
        self.jumping = self.falling = False
        self.on_bridge = above_bridge

    if tap and not self.jumping:
      self.jumping, self.on_bridge = True, False
      self.vy = -2
      self.multiplier = math.ceil(self.multiplier / 2)
    if not self.jumping and self.y > BRIDGE_TOP and above_bridge:
      self.terminal = True

    if self.on_bridge:
      self.score_timer -= 1
      if self.score_timer < 0:
        reward = self.multiplier
        self.multiplier += 1
        self.score_timer = 10
        return reward
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(self.y) // 5
    state[row, :3, 0] = True
    state[row, :3, 3] = self.on_bridge
    state[8:, :, 2] = True
    for bridge in self.bridges:
      for column in range(10):
        state[7, column, 1] |= column * 15 < bridge.x + bridge.length and (column + 1) * 15 > bridge.x
    state[0, :3, 4] = self.pressed, self.jumping, self.vy < 0
    for row, value in enumerate((self.steps, self.multiplier, self.score_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
