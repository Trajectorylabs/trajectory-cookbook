from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Actor:
  x: float
  y: float
  vx: float = 0.0
  vy: float = 0.0
  target_x: float = 50.0
  target_y: float = 50.0

  def distance(self, other: "Actor") -> float:
    return math.hypot(self.x - other.x, self.y - other.y)

  def steer(self, angle: float, force: float, damping: float) -> None:
    self.vx = (self.vx + math.cos(angle) * force) * damping
    self.vy = (self.vy + math.sin(angle) * force) * damping
    self.x += self.vx
    self.y += self.vy


@dataclass
class Explosion:
  x: float
  y: float
  radius: float = 1.0
  growing: bool = True

  def hits(self, actor: Actor) -> bool:
    return abs(math.hypot(self.x - actor.x, self.y - actor.y) - self.radius) < 4


class InvincibleManEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = Actor(50, 40)
    self.direction = -1
    self.humans = [self._human(*self.random.uniform(40, 60, 2)) for _ in range(10)]
    self.enemies: list[Actor] = []
    self.explosions: list[Explosion] = []
    self.human_timer = self.enemy_timer = 0.0
    self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False

  def _human(self, x: float, y: float) -> Actor:
    tx, ty = self.random.uniform(40, 60, 2)
    return Actor(x, y, target_x=tx, target_y=ty)

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction *= -1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    player = self.player
    toward = math.atan2(50 - player.y, 50 - player.x)
    angle, speed = (toward, -1) if self.pressed else (toward + math.pi / 2 * self.direction, 0.7)
    player.x += math.cos(angle) * speed
    player.y += math.sin(angle) * speed
    pull = (math.hypot(player.x - 50, player.y - 50) + 9) * 0.005
    player.x = min(100, max(0, player.x + math.cos(toward) * pull))
    player.y = min(100, max(0, player.y + math.sin(toward) * pull))
    for blast in self.explosions:
      blast.radius += 0.6 if blast.growing else -0.6
      if blast.radius > 12:
        blast.growing = False
    self.explosions = [blast for blast in self.explosions if blast.radius >= 1]
    self.human_timer -= 1
    if self.human_timer < 0:
      if len(self.humans) < 9:
        count = len(self.humans) + 1
        self.humans.append(self._human((50 + sum(h.x for h in self.humans)) / count,
                                       (50 + sum(h.y for h in self.humans)) / count))
      self.human_timer = 600
    self._move_humans()
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      angle = self.random.uniform(0, math.tau)
      x, y = 50 + 80 * math.cos(angle), 50 + 80 * math.sin(angle)
      count = int(self.random.integers(3, 9))
      for _ in range(count):
        dx, dy = self.random.uniform(-9, 9, 2)
        vx, vy = self.random.uniform(-1, 1, 2)
        self.enemies.append(Actor(x + dx, y + dy, vx, vy))
      self.enemies = self.enemies[-64:]
      self.enemy_timer = self.random.uniform(70, 99) * math.sqrt(count)
    reward = self._move_enemies()
    if not self.explosions:
      self.multiplier = 1
    self.terminal = not self.humans
    return reward

  def _move_humans(self) -> None:
    alive = []
    for human in self.humans:
      enemy = min(self.enemies, key=human.distance, default=None)
      if enemy is not None and human.distance(enemy) < 25:
        angle = math.atan2(human.y - enemy.y, human.x - enemy.x)
      else:
        if math.hypot(human.x - human.target_x, human.y - human.target_y) < 1:
          human.target_x, human.target_y = self.random.uniform(40, 60, 2)
        angle = math.atan2(human.target_y - human.y, human.target_x - human.x)
      human.steer(angle, 0.01, 0.9)
      human.x = min(90, max(10, human.x))
      human.y = min(90, max(10, human.y))
      if not any(blast.hits(human) for blast in self.explosions):
        alive.append(human)
    self.humans = alive

  def _move_enemies(self) -> int:
    reward = 0
    alive = []
    for enemy in self.enemies:
      human = min(self.humans, key=enemy.distance, default=None)
      if human is not None:
        enemy.steer(math.atan2(human.y - enemy.y, human.x - enemy.x), 0.005, 0.95)
      if (enemy.distance(self.player) < 5 or any(enemy.distance(h) < 5 for h in self.humans)
          or any(blast.hits(enemy) for blast in self.explosions)):
        self.explosions.append(Explosion(enemy.x, enemy.y))
        self.explosions = self.explosions[-16:]
        reward += self.multiplier
        self.multiplier += 1
      else:
        alive.append(enemy)
    self.enemies = alive
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, actors in ((0, [self.player]), (1, self.humans), (2, self.enemies)):
      for actor in actors:
        if 0 <= actor.x <= 100 and 0 <= actor.y <= 100:
          state[min(9, int(actor.y) // 10), min(9, int(actor.x) // 10), channel] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for blast in self.explosions:
      distance = np.hypot(cols - blast.x, rows - blast.y)
      state[:, :, 3 if blast.growing else 4] |= abs(distance - blast.radius) < 6
    for row, value in enumerate((self.steps, len(self.humans), min(511, self.multiplier))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
