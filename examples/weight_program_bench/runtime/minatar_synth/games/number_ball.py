from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Floor:
  x: float
  width: float
  value: int = 0


@dataclass
class Ball:
  value: int
  x: float = 10
  y: float = 47
  vx: float = 0
  vy: float = 0
  mode: str = "ready"


class NumberBallEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.floors = [Floor(0, 200)]
    self.balls: list[Ball] = []
    self.frames = self.steps = 0
    self.boost = self.launch_penalty = 0.0
    self.angle = -0.1
    self.multiplier = 1
    self.add_ball = True
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    release = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, release):
    lead = self.floors[0]
    scroll = 0.04 + self.boost + max(0, lead.x + lead.width - 100) * 0.1
    self.boost *= 0.9
    self.launch_penalty = min(9, max(0, self.launch_penalty - 0.01))
    reward = self._move_floors(scroll)
    if self.terminal:
      return reward
    if self.frames > 30 and self.add_ball:
      values = {floor.value for floor in self.floors}
      for _ in range(9):
        value = int(self.random.integers(1, 3))
        if value in values:
          break
      self.balls.append(Ball(value))
      self.add_ball = False
      self.angle = -0.1
    remaining = []
    for ball in self.balls:
      self._move_ball(ball, release, scroll)
      if -20 <= ball.x <= 220 and -50 <= ball.y <= 63:
        remaining.append(ball)
      elif ball.mode == "flying":
        self.add_ball = True
    self.balls = remaining
    self.frames += 1
    return reward

  def _move_floors(self, scroll):
    reward = 0
    right = 0
    remaining = []
    for index, floor in enumerate(self.floors):
      floor.x -= scroll
      right = floor.x + floor.width
      if floor.value and any(ball.mode == "floor" and ball.value == floor.value
                             and floor.x - 3 < ball.x < right + 3 for ball in self.balls):
        reward += floor.value * self.multiplier
        self.multiplier = min(16, self.multiplier * 2)
        continue
      if index + 1 < len(self.floors):
        gap = self.floors[index + 1].x - right
        floor.x += gap * 0.1 if gap > -2 else gap
      remaining.append(floor)
    self.floors = remaining
    self.terminal = not remaining or remaining[0].x + remaining[0].width < 30
    if right < 200:
      self.floors.append(Floor(200, self.random.uniform(20, 50), int(self.random.integers(1, 3))))
    return reward

  def _move_ball(self, ball, release, scroll):
    if ball.mode == "ready":
      self.angle -= 0.01 * self.pressed
      if release or self.angle < -math.pi * 0.47:
        self.multiplier = max(1, self.multiplier // 2)
        self.boost = self.launch_penalty
        self.launch_penalty += 1
        ball.vx, ball.vy = 5 * math.cos(self.angle), 5 * math.sin(self.angle)
        ball.mode = "flying"
    elif ball.mode == "flying":
      ball.vx *= 0.99
      ball.vy = (ball.vy + 0.1) * 0.99
      ball.x += ball.vx
      ball.y += ball.vy
      if ball.y > 47:
        ball.y = 47
        ball.vx *= 0.5
        ball.vy *= -0.5
        if abs(ball.vy) < 0.5:
          ball.mode = "floor"
          self.add_ball = True
    elif ball.mode == "floor":
      ball.vx *= 0.9
      ball.x += ball.vx
      ball.vy = 0
      if ball.x < 30:
        ball.mode = "falling"
    else:
      ball.vy += 0.1
      ball.y += ball.vy
    if ball.mode in ("flying", "floor"):
      ball.x -= scroll
      for other in self.balls:
        dx, dy = ball.x - other.x, ball.y - other.y
        distance = math.hypot(dx, dy)
        if other is not ball and distance < 6:
          angle = math.atan2(dy, dx)
          ux, uy = math.cos(angle), math.sin(angle)
          push = math.hypot(other.vx, other.vy) * 0.7
          ball.vx += ux * push
          ball.vy += uy * push
          push = math.hypot(ball.vx, ball.vy) * 0.7
          other.vx -= ux * push
          other.vy -= uy * push

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for ball in self.balls:
      if 0 <= ball.x < 200 and 0 <= ball.y < 60:
        row, col = int(ball.y // 6), int(ball.x // 20)
        state[row, col, ball.value - 1] = True
        state[row, col, 4] |= ball.mode == "ready"
    for floor in self.floors:
      left, right = max(0, int(floor.x // 20)), min(9, int((floor.x + floor.width) // 20))
      if left <= right:
        state[9, left:right + 1, 2:4] |= [floor.value != 2, floor.value != 1]
    for row, value in enumerate((self.steps, self.multiplier, round(-self.angle * 100),
                                 round(self.boost * 40))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
