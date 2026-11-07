from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Hole:
  x: float


@dataclass
class Walker:
  x: float
  y: float
  mode: str = "walk"
  hole: Hole | None = None
  dot_timer: float = 0.0


class JumpOnEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = Walker(5, 87)
    self.holes = [Hole(30), Hole(60)]
    self.enemies: list[Walker] = []
    self.dots: list[tuple[float, float]] = []
    self.hole_distance = self.group_timer = self.enemy_timer = 0.0
    self.group_size = self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False

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
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    scroll = 0.03 + max(0, self.player.x - 15) * 0.1
    self.hole_distance -= scroll
    if self.hole_distance < 0:
      self.holes.append(Hole(104))
      self.hole_distance += self.random.uniform(30, 45)
    for hole in self.holes:
      hole.x -= scroll
    self.holes = [hole for hole in self.holes if hole.x >= -4]
    self.group_timer -= 1
    if self.group_timer < 0:
      self.group_size = int(self.random.integers(1, 3))
      self.group_timer = self.group_size * 8 + self.random.uniform(100, 110)
      self.enemy_timer = 0
    self.enemy_timer -= 1
    if self.group_size and self.enemy_timer < 0:
      self.enemies.append(Walker(103, 87 - 12 * self.random.integers(6),
                                 dot_timer=self.random.uniform(0, 6)))
      self.group_size -= 1
      self.enemy_timer = self.random.uniform(8, 10)
    walking_enemies = []
    for enemy in self.enemies:
      was_walking = enemy.mode == "walk"
      leave = -9 < enemy.y - 1 - self.player.y < 1
      self._move(enemy, -1, scroll, leave)
      if was_walking:
        walking_enemies.append(enemy)
      if enemy.mode == "walk":
        enemy.dot_timer -= 1
        if enemy.dot_timer < 0:
          self.dots.append((enemy.x, enemy.y))
          enemy.dot_timer += 6
    self.enemies = [enemy for enemy in self.enemies if enemy.x >= -3]
    was_walking = self.player.mode == "walk"
    self._move(self.player, 1, scroll, self.pressed)
    self.terminal = self.player.x < 0 or (was_walking and any(
      abs(enemy.x - self.player.x) < 5 and abs(enemy.y - self.player.y) < 5
      for enemy in walking_enemies
    ))
    reward = 0
    remaining = []
    for x, y in self.dots:
      x -= scroll
      if self.player.mode not in ("up", "down") and abs(x - self.player.x) < 4 and abs(y - self.player.y) < 4:
        reward += self.multiplier
        self.multiplier += 1
      elif x >= -3:
        remaining.append((x, y))
    self.dots = remaining
    return reward

  def _move(self, walker: Walker, direction: int, scroll: float, leave: bool) -> None:
    if walker.mode == "walk":
      walker.x += direction - scroll
      ahead = [hole for hole in self.holes if 0 < (hole.x - walker.x) * direction < 8]
      if ahead:
        walker.hole = min(ahead, key=lambda hole: abs(hole.x - walker.x))
        walker.mode = "enter"
      return
    hole = walker.hole
    assert hole is not None
    if walker.mode == "enter":
      walker.x += direction - scroll
      distance = (hole.x - walker.x) * direction
      walker.y += 1 if distance < 4 else -1
      if distance <= 0:
        walker.mode = "down"
    elif walker.mode == "down":
      walker.x = hole.x
      walker.y += 1
      if walker.y > 90:
        walker.mode = "up"
    elif walker.mode == "up":
      walker.x = hole.x
      walker.y -= 1
      if walker.y < 23:
        walker.mode = "down"
      if leave:
        walker.mode = "exit"
        walker.y = math.ceil((walker.y - 90) / 12) * 12 + 87
    else:
      walker.x += direction - scroll
      distance = (walker.x - hole.x) * direction
      walker.y += -1 if distance < 4 else 1
      if distance >= 8:
        walker.mode = "walk"
        walker.y = math.floor((walker.y - 90) / 12 + 0.5) * 12 + 87
        if direction == -1:
          self.multiplier = 1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.player.y // 10))),
          min(9, max(0, int(self.player.x // 10))), 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 100 and 0 <= enemy.y < 100:
        state[int(enemy.y // 10), int(enemy.x // 10), 1] = True
    for x, y in self.dots:
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 2] = True
    for hole in self.holes:
      if 0 <= hole.x < 100:
        state[2:10, int(hole.x // 10), 3] = True
    for y in range(90, 29, -12):
      state[y // 10, :, 4] = True
    mode = ("walk", "enter", "down", "up", "exit").index(self.player.mode)
    for row, value in enumerate((self.steps, min(511, self.multiplier), mode, round(self.player.y * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
