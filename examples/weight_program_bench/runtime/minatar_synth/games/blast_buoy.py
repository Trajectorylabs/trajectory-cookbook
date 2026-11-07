from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
BLAST_RADIUS = 25


@dataclass
class Submarine:
  x: float
  y: float
  vx: float
  fire_timer: int


@dataclass
class Bullet:
  x: float
  y: float
  vx: float


class BlastBuoyEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = self.combo_timer = self.shield = self.freeze = 0
    self.combo = 1
    self.pressed = self.terminal = False
    self.x, self.y = 30.0, 40.0
    self.vx = self.vy = 0.0
    self.submarines: list[Submarine] = []
    self.bullets: list[Bullet] = []

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

  def _advance_frame(self) -> int:
    if self.freeze:
      self.freeze -= 1
      return 0
    self.vy = min(1.1, max(-1.1, self.vy + (0.06 if self.pressed else -0.05)))
    self.vx *= 0.95
    self.x = min(95, max(5, self.x + self.vx))
    self.y = min(87, max(13, self.y + self.vy))
    if self.y in (13, 87):
      self.vy = 0

    self.shield = max(0, self.shield - 1)
    if self.combo_timer:
      self.combo_timer -= 1
    else:
      self.combo = 1

    if not self.spawn_timer:
      direction = int(self.random.choice((-1, 1)))
      self.submarines.append(Submarine(
        -8 if direction > 0 else 108,
        self.random.uniform(13, 87),
        direction * self.random.uniform(0.2, 0.45),
        int(self.random.integers(60, 121)),
      ))
      self.spawn_timer = int(self.random.integers(50, 91))
    self.spawn_timer -= 1

    contact = None
    for sub in self.submarines:
      sub.x += sub.vx
      sub.fire_timer -= 1
      if sub.fire_timer < 0 and 0 < sub.x < 100 and abs(sub.y - self.y) < 10:
        if math.hypot(sub.x - self.x, sub.y - self.y) > 36:
          direction = 1 if self.x > sub.x else -1
          self.bullets.append(Bullet(sub.x + direction * 6, sub.y, direction * 0.95))
        sub.fire_timer = int(self.random.integers(90, 151))
      if contact is None and not self.shield and abs(sub.x - self.x) < 7.5 and abs(sub.y - self.y) < 5:
        contact = sub

    self.submarines = [sub for sub in self.submarines if -12 <= sub.x <= 112]
    reward = self._blast(contact) if contact is not None else 0

    survivors = []
    for bullet in self.bullets:
      bullet.x += bullet.vx
      if abs(bullet.x - self.x) < 5.5 and abs(bullet.y - self.y) < 4:
        self.terminal |= not self.shield
      elif -5 <= bullet.x <= 105:
        survivors.append(bullet)
    self.bullets = survivors
    return reward

  def _blast(self, center: Submarine) -> int:
    self.submarines.remove(center)
    remaining = [
      sub for sub in self.submarines
      if math.hypot(sub.x - center.x, sub.y - center.y) >= BLAST_RADIUS
    ]
    extras = len(self.submarines) - len(remaining)
    self.submarines = remaining
    self.bullets = [
      bullet for bullet in self.bullets
      if math.hypot(bullet.x - center.x, bullet.y - center.y) >= BLAST_RADIUS
    ]
    reward = (5 + 8 * extras) * self.combo
    self.combo += 1
    self.combo_timer = 120
    angle = math.atan2(self.y - center.y, self.x - center.x)
    self.vx, self.vy = 2.2 * math.cos(angle), 2.2 * math.sin(angle)
    self.shield = 40
    self.freeze = min(8, 3 + extras)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, column = int(self.y) // 10, int(self.x) // 10
    state[row, column, 0] = True
    state[row, column, 4] = self.shield > 0
    for sub in self.submarines:
      if 0 <= sub.x < 100:
        state[int(sub.y) // 10, int(sub.x) // 10, 1 if sub.vx > 0 else 2] = True
    for bullet in self.bullets:
      if 0 <= bullet.x < 100:
        state[int(bullet.y) // 10, int(bullet.x) // 10, 3] = True
    for row, value in enumerate((self.steps, self.combo, self.combo_timer, self.shield, self.freeze)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.vy > 0
    return state
