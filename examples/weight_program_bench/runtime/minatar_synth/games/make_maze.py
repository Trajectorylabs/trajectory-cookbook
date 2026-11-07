from collections import deque
from dataclasses import dataclass

import numpy as np


_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class Enemy:
  row: int
  col: int
  turn: int
  heading: int = 1
  timer: int = 0
  angry: bool = False


class MakeMazeEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.walls = np.zeros((10, 10), dtype=bool)
    self.walls[:, (0, 9)] = True
    self.walls[5, (1, 8)] = True
    self.walls[6, 2:8] = True
    self.walls[7, 1:4] = self.walls[7, 6:9] = True
    self.cursor = (5, 5)
    self.enemies = []
    self.gold = set()
    self.scroll = 10.
    self.penalty = 0.
    self.timer = self.steps = self.action = 0
    self.multiplier = 1
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    if self.action in (1, 2, 3, 4):
      dr, dc = _DIRECTIONS[(self.action + 1) % 4]
      row, col = self.cursor
      self.cursor = min(9, max(0, row + dr)), min(8, max(1, col + dc))
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(frame == 0 and self.action == 5 and previous != 5)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, toggle):
    oldest = min((r for r, _ in self.gold), default=10)
    self.scroll += 0.01 + self.penalty + max(0, oldest * 10 - self.scroll - 50) * 0.02
    self.penalty *= 0.9
    if self.scroll >= 10:
      self.scroll -= 10
      self.walls[:-1] = self.walls[1:].copy()
      self.walls[-1, 1:9] = False
      self.gold = {(row - 1, col) for row, col in self.gold}
      self.gold.add((9, int(self.random.integers(1, 9))))
      for enemy in self.enemies:
        enemy.row -= 1
    if toggle and self.cursor not in self.gold:
      self.walls[self.cursor] = not self.walls[self.cursor]
    self.timer -= 1
    if self.timer < 0:
      columns = [col for col in range(1, 9) if not self.walls[0, col]]
      col = int(self.random.choice(columns or range(1, 9)))
      self.enemies.append(Enemy(0, col, int(self.random.choice((-1, 1)))))
      self.timer = 150

    reward = 0
    remaining = []
    for enemy in self.enemies:
      if enemy.row < 0:
        continue
      enemy.timer -= 1
      if enemy.timer < 0:
        reward += self._collect((enemy.row, enemy.col))
        reward += self._move(enemy)
        enemy.timer = 60
      if enemy.row >= 10:
        self.penalty += 1
        self.multiplier = max(1, self.multiplier - 1)
        reward -= self.multiplier
      elif enemy.row >= 0:
        remaining.append(enemy)
    self.enemies = remaining
    self.terminal = any(row < 0 for row, _ in self.gold)
    return reward

  def _collect(self, cell):
    if cell not in self.gold:
      return 0
    self.gold.remove(cell)
    self.multiplier += 1
    return self.multiplier - 1

  def _move(self, enemy):
    cell = enemy.row, enemy.col
    enemy.angry = not self._has_exit(cell)
    if enemy.angry:
      enemy.heading = 1
      enemy.row += 1
      if enemy.row < 10:
        self.walls[enemy.row, enemy.col] = False
      return self._collect((enemy.row, enemy.col))
    reward = 0
    # Follow the chosen wall, running through straight corridors in one move.
    for index in range(100):
      turns = (enemy.turn, 0, -enemy.turn, 2)
      choices = ((enemy.heading + turn) % 4 for turn in turns)
      heading = next((h for h in choices if self._open(_step(cell, h))), None)
      if heading is None or (index and heading != enemy.heading):
        break
      enemy.heading = heading
      cell = _step(cell, heading)
      enemy.row, enemy.col = cell
      reward += self._collect(cell)
      if not 0 <= enemy.row < 10:
        break
    return reward

  def _open(self, cell):
    row, col = cell
    return 0 < col < 9 and (not 0 <= row < 10 or not self.walls[cell])

  def _has_exit(self, start):
    pending = deque([start])
    seen = {start}
    while pending:
      cell = pending.popleft()
      if cell[0] == 9:
        return True
      for heading in range(4):
        other = _step(cell, heading)
        if other[0] >= 0 and other not in seen and self._open(other):
          seen.add(other)
          pending.append(other)
    return False

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[:, :, 0] = self.walls
    for row, col in self.gold:
      if row >= 0:
        state[row, col, 1] = True
    for enemy in self.enemies:
      state[enemy.row, enemy.col, 2 + enemy.angry] = True
    state[self.cursor[0], self.cursor[1], 4] = True
    for row, value in enumerate((self.steps, self.action, self.multiplier, round(self.scroll * 20), self.timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _step(cell, heading):
  dr, dc = _DIRECTIONS[heading]
  return cell[0] + dr, cell[1] + dc
