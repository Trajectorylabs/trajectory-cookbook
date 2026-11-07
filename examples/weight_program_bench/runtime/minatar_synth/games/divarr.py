from dataclasses import dataclass
import math

import numpy as np


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Shot:
  x: float = 50
  y: float = 85
  heading: int = 3
  overlap: int = 0


@dataclass
class Falling:
  x: float
  y: float
  vx: float
  vy: float
  human: bool


class DivarrEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.until_human = 3
    self.first_human = True
    self.pressed = self.terminal = False
    self.shots: list[Shot] = []
    self.falling: list[Falling] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self._fire_or_split()
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _fire_or_split(self) -> None:
    if not self.shots:
      self.shots = [Shot()]
      return
    offsets = (-1, 1) if len(self.shots) < 16 else (1,)
    self.shots = [Shot(shot.x, shot.y, (shot.heading + offset) % 4)
                  for shot in self.shots for offset in offsets]

  def _advance_frame(self) -> int:
    visible = []
    for shot in self.shots:
      dx, dy = DIRECTIONS[shot.heading]
      shot.x += dx
      shot.y += dy
      if -2 <= shot.x < 102 and -2 <= shot.y < 88:
        touches = any(abs(other.x - shot.x) < 3 and abs(other.y - shot.y) < 3 for other in visible)
        shot.overlap = shot.overlap + 1 if touches else 0
        if shot.overlap <= 8:
          visible.append(shot)
    self.shots = visible

    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self._spawn()
    reward = 0
    remaining = []
    for obj in self.falling:
      obj.x += obj.vx
      obj.y += obj.vy
      hittable = not obj.human or obj.y >= 20
      if hittable and any(abs(obj.x - shot.x) < 4 and abs(obj.y - shot.y) < 4 for shot in self.shots):
        if obj.human:
          self.terminal = True
        else:
          reward += len(self.shots)
      elif obj.y > 88:
        if obj.human:
          reward += 100
        else:
          self.terminal = True
      else:
        remaining.append(obj)
    self.falling = remaining
    return reward

  def _spawn(self) -> None:
    self.until_human -= 1
    human = self.until_human < 0
    x = self.random.uniform(9, 90)
    if human:
      if self.first_human:
        x = 25
        self.first_human = False
      self.until_human = int(self.random.integers(3, 15))
    dx = min(90, max(9, x + self.random.uniform(-10, 10))) - x
    speed = self.random.uniform(0.1, 0.2) * (1.6 if human else 1)
    distance = math.hypot(dx, 93)
    self.falling.append(Falling(x, -3, dx * speed / distance, 93 * speed / distance, human))
    self.spawn_timer = int(self.random.integers(60, 81))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, 5, 0] = True
    state[9, :, 4] = True
    for shot in self.shots:
      if 0 <= shot.x < 100 and 0 <= shot.y < 100:
        state[int(shot.y) // 10, int(shot.x) // 10, 1] = True
    for obj in self.falling:
      if 0 <= obj.y < 100:
        state[int(obj.y) // 10, int(obj.x) // 10, 3 if obj.human else 2] = True
    for row, value in enumerate((self.steps, len(self.shots), self.spawn_timer, self.until_human + 1)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
