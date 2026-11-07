from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float
  vy: float
  size: float

  def move(self) -> None:
    self.x += self.vx
    self.y += self.vy
    if self.x > 95 or self.x < 5:
      self.vx *= -1


@dataclass
class Bubble:
  x: float
  y: float
  size: float
  life: int = 180


@dataclass
class Wall:
  x: float
  y: float
  width: float

  def bounce(self, body: Body, damping: float) -> None:
    clearance = (body.size + 4) / 2
    if abs(body.x - self.x) < (self.width + body.size) / 2 and abs(body.y - self.y) < clearance:
      body.vy *= -damping
      body.y = self.y + (clearance + 1) * (-1 if body.y < self.y else 1)


class GravityBubbleEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = Body(50, 50, 1, 0, 6)
    self.height = 0.0
    self.steps = self.frames = self.best_height = self.hold = 0
    self.next_wall = 150.0
    self.next_enemy = 60.0
    self.pressed = self.terminal = False
    self.preview = Bubble(50, 50, 8)
    self.bubbles: list[Bubble] = []
    self.enemies: list[Body] = []
    self.walls: list[Wall] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, released: bool) -> int:
    player = self.player
    player.vy = (player.vy + 0.025) * 0.995
    player.move()
    scroll = max(0, (60 - player.y) * 0.5)
    player.y += scroll
    self.height += scroll
    for objects in (self.bubbles, self.enemies, self.walls):
      for obj in objects:
        obj.y += scroll
    if self.pressed:
      self.hold += 1
      x = player.x
      vx = player.vx
      for _ in range(self.hold):
        x += vx
        if x > 95 or x < 5:
          vx *= -1
      self.preview = Bubble(x, player.y + player.vy * self.hold, 8 + min(25, self.hold * 0.2))
    else:
      if released:
        self.bubbles.append(self.preview)
      self.hold = 0
    altitude = self.height + max(0, 100 - player.y)
    self._spawn(altitude)
    self.frames += 1
    for bubble in self.bubbles:
      bubble.life -= 1
      radius = bubble.size * min(1, bubble.life / 150)
      if math.hypot(player.x - bubble.x, player.y - bubble.y) < radius + 9:
        player.vy -= 0.05
    reward = max(0, int(altitude) - self.best_height)
    self.best_height = max(self.best_height, int(altitude))
    surviving = []
    for enemy in self.enemies:
      inside = False
      for bubble in self.bubbles:
        radius = bubble.size * min(1, bubble.life / 150)
        inside = math.hypot(enemy.x - bubble.x, enemy.y - bubble.y) < radius + enemy.size / 2
        if inside:
          enemy.vy -= 0.08
      if not inside:
        enemy.vy += 0.02
      enemy.move()
      for wall in self.walls:
        wall.bounce(enemy, 0.6)
      if math.hypot(player.x - enemy.x, player.y - enemy.y) < (6 + enemy.size) / 2 + 2:
        player.vy += enemy.vy * 1.5
        reward += 100
      elif enemy.y <= 110:
        surviving.append(enemy)
    self.enemies = surviving
    self.bubbles = [bubble for bubble in self.bubbles if bubble.life]
    for wall in self.walls:
      wall.bounce(player, 0.8)
    self.walls = [wall for wall in self.walls if wall.y <= 110]
    self.terminal = player.y > 105
    return reward

  def _spawn(self, altitude: float) -> None:
    level = int(altitude // 100) + 1
    if self.frames >= self.next_enemy:
      speed = self.random.uniform(0.2, 0.25 + math.sqrt(level) * 0.08)
      self.enemies.append(Body(self.random.uniform(15, 85), -5, self.random.uniform(-0.6, 0.6),
                               speed, 7 + min(3, level // 2)))
      interval = max(45, 120 - level * 10)
      self.next_enemy = self.frames + interval * self.random.uniform(0.7, 1.3)
    if level > 2 and altitude >= self.next_wall:
      width = self.random.uniform(25, 50)
      self.walls.append(Wall(self.random.uniform(width / 2, 100 - width / 2), -5, width))
      self.next_wall = altitude + self.random.uniform(70, 150)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    player = self.player
    state[min(9, max(0, int(player.y) // 10)), int(player.x) // 10, 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.y < 100:
        state[int(enemy.y) // 10, int(enemy.x) // 10, 1] = True
    for wall in self.walls:
      if 0 <= wall.y < 100:
        state[int(wall.y) // 10, int(wall.x - wall.width / 2) // 10:
              int(wall.x + wall.width / 2) // 10 + 1, 2] = True
    for channel, bubbles in ((3, self.bubbles), (4, [self.preview] if self.pressed else [])):
      for bubble in bubbles:
        radius = bubble.size * min(1, bubble.life / 150)
        rows, cols = np.ogrid[5:100:10, 5:100:10]
        state[:, :, channel] |= (cols - bubble.x) ** 2 + (rows - bubble.y) ** 2 < radius ** 2
    velocity = min(511, max(0, round((player.vy + 5) * 50)))
    for row, value in enumerate((self.steps, min(511, self.hold), velocity)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, player.vx > 0
    return state
