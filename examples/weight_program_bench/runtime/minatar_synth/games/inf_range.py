from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Arrow:
  x: float
  y: float
  angle: float


class InfRangeEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 80.0
    self.angle = self.tap_angle = -math.pi / 2
    self.scroll_x = self.scroll_y = 0.0
    self.steps = self.frames = 0
    self.tap_frame = -99
    self.spawn_timer = 200.0
    self.multiplier = 10.9
    self.pressed = self.terminal = False
    self.arrows: list[Arrow] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap, release = pressed and not self.pressed, self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool, release: bool) -> int:
    self.x += math.cos(self.angle)
    self.y += math.sin(self.angle)
    if not 40 <= self.x <= 60:
      self.scroll_x = (min(60, max(40, self.x)) - self.x) * 0.2
    if not 40 <= self.y <= 60:
      self.scroll_y = (min(60, max(40, self.y)) - self.y) * 0.2
    self.x += self.scroll_x
    self.y += self.scroll_y
    drawn_angle = self.angle
    if tap:
      self.tap_angle, self.tap_frame = self.angle, self.frames
    if self.pressed:
      self.angle += 0.02
    if release and self.frames - self.tap_frame < 9:
      self.angle = self.tap_angle + math.pi / 2
    self.frames += 1
    reward = 0
    remaining = []
    for arrow in self.arrows:
      arrow.x = (arrow.x + math.cos(arrow.angle) * 0.5 + self.scroll_x + 3) % 106 - 3
      arrow.y = (arrow.y + math.sin(arrow.angle) * 0.5 + self.scroll_y + 3) % 106 - 3
      dx, dy = arrow.x - self.x, arrow.y - self.y
      arrow_tip = (dx + 4 * math.cos(arrow.angle), dy + 4 * math.sin(arrow.angle))
      player_tip = (4 * math.cos(drawn_angle), 4 * math.sin(drawn_angle))
      body_hit = math.hypot(dx, dy) < 3 or math.dist((dx, dy), player_tip) < 3
      tip_hit = math.hypot(*arrow_tip) < 3
      if body_hit or tip_hit:
        facing = abs((self.angle - arrow.angle) % math.tau - math.pi) < 0.7
        if facing:
          arrow.angle += math.pi
          arrow.x += math.cos(arrow.angle)
          arrow.y += math.sin(arrow.angle)
          self.angle += math.pi
        elif body_hit:
          reward += int(self.multiplier)
          self.multiplier += 10
          continue
        else:
          self.terminal = True
      remaining.append(arrow)
    self.arrows = remaining
    self.spawn_timer = self.spawn_timer - 1 if remaining else -1
    if self.spawn_timer < 0:
      self._spawn()
    self.multiplier = max(10.9, self.multiplier - 0.02)
    return reward

  def _spawn(self) -> None:
    for _ in range(9):
      x, y = self.random.uniform(10, 90, 2)
      dx, dy = (self.x - x + 50) % 100 - 50, (self.y - y + 50) % 100 - 50
      if abs(dx) + abs(dy) > 36:
        self.arrows.append(Arrow(x, y, self.random.uniform(0, math.tau)))
        self.spawn_timer = self.random.uniform(150, 250)
        break

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for x, y, channel in ((self.x, self.y, 0),
                          (self.x + 8 * math.cos(self.angle), self.y + 8 * math.sin(self.angle), 1)):
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, channel] = True
    for arrow in self.arrows:
      for x, y, channel in ((arrow.x, arrow.y, 2),
                            (arrow.x + 4 * math.cos(arrow.angle), arrow.y + 4 * math.sin(arrow.angle), 3),
                            (arrow.x + 10 * math.cos(arrow.angle), arrow.y + 10 * math.sin(arrow.angle), 4)):
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, channel] = True
    values = (self.steps, min(511, int(self.multiplier)), round(self.angle % math.tau * 80),
              min(9, self.frames - self.tap_frame))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
