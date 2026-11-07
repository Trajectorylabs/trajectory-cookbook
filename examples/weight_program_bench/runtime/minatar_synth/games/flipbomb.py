from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float
  vy: float
  launched: bool = False

  def move(self) -> None:
    self.x += self.vx
    self.y += self.vy


@dataclass
class Explosion:
  x: float
  y: float
  angle: float
  age: int = 0

  def lobes(self):
    radius = self.age * 0.5 if self.age < 20 else 30 - self.age
    half_width = 5 if not self.age else (radius + 3) / 2
    for index in range(5):
      angle = self.angle + index * math.tau / 5
      yield self.x + radius * math.cos(angle), self.y + radius * math.sin(angle), half_width


class FlipbombEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.ball_timer = 0
    self.bomb_timer = 0.0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.balls: list[Body] = []
    self.bombs: list[Body] = []
    self.explosions: list[Explosion] = []

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
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool) -> int:
    if tap:
      self.multiplier = 1
    lobes = [lobe for explosion in self.explosions for lobe in explosion.lobes()]
    for explosion in self.explosions:
      explosion.age += 1
      explosion.angle += 0.2
    self.explosions = [explosion for explosion in self.explosions if explosion.age < 30]
    self.ball_timer -= 1
    if self.ball_timer < 0:
      self.balls.append(Body(99, 70, -1, 0.5))
      self.ball_timer = 60
    balls = []
    for ball in self.balls:
      ball.move()
      if tap and not ball.launched and ball.x < 70:
        angle = (ball.x - 70) * 0.06
        ball.vx = 2 * math.cos(angle) + 4 * math.sin(angle)
        ball.vy = 2 * math.sin(angle) - 4 * math.cos(angle)
        ball.launched = True
      if 0 <= ball.x < 99 and 0 <= ball.y < 99 and not self._blast_hit(ball, lobes, 1.5):
        balls.append(ball)
    self.balls = balls

    self.bomb_timer -= 1
    if self.bomb_timer < 0:
      x = self.random.uniform(0, 80)
      self.bombs.append(Body(x, 0, (self.random.uniform(20, 70) - x) / 500, 0.14))
      self.bomb_timer += self.random.uniform(30, 50)

    reward = 0
    bombs = []
    for bomb in self.bombs:
      bomb.move()
      if self._blast_hit(bomb, lobes, 2.5) or any(
        ball.launched and abs(ball.x - bomb.x) < 4 and abs(ball.y - bomb.y) < 4
        for ball in self.balls
      ):
        self.explosions.append(Explosion(bomb.x, bomb.y, self.random.uniform(-math.pi, math.pi)))
        reward += self.multiplier
        self.multiplier += 1
      elif bomb.y > 99 or any(
        not ball.launched and abs(ball.x - bomb.x) < 4 and abs(ball.y - bomb.y) < 4
        for ball in self.balls
      ) or self._hits_ramp(bomb, tap):
        self.terminal = True
      else:
        bombs.append(bomb)
    self.bombs = bombs
    return reward

  @staticmethod
  def _blast_hit(body: Body, lobes: list[tuple[float, float, float]], margin: float) -> bool:
    return any(abs(body.x - x) < radius + margin and abs(body.y - y) < radius + margin
               for x, y, radius in lobes)

  @staticmethod
  def _hits_ramp(bomb: Body, tap: bool) -> bool:
    return (
      72 <= bomb.x <= 102 and abs(bomb.y - (75 + (99 - bomb.x) * 11 / 24)) < 3
      or 47 <= bomb.x <= 73 and abs(bomb.y - (89 + (70 - bomb.x) * (-0.45 if tap else 0.45))) < 3
    )

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for ball in self.balls:
      state[int(ball.y) // 10, int(ball.x) // 10, int(ball.launched)] = True
    for bomb in self.bombs:
      if 0 <= bomb.x < 100 and 0 <= bomb.y < 100:
        state[int(bomb.y) // 10, int(bomb.x) // 10, 2] = True
    for explosion in self.explosions:
      for x, y, _ in explosion.lobes():
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, 3] = True
    state[8, 5:8, 4] = True
    state[7:9, 8:10, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.ball_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
