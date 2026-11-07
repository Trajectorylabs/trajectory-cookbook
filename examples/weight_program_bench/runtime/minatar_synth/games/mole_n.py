from dataclasses import dataclass

import numpy as np


@dataclass
class Mole:
  x: int
  y: int
  value: int
  timer: int = 0
  award: int = 0


class MoleNEnvironment:
  score_cap = 40
  moves = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))
  mole_moves = ((1, 0), (0, 1), (-1, 0), (0, -1))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.numbers = self.random.integers(0, 10, (9, 9))
    self.x, self.y = 4, 4
    self.target = 1
    self.scroll = self.penalty = self.spawn_distance = 0.0
    self.target_count = self.move_timer = self.steps = self.action = 0
    self.moles: list[Mole] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    dx, dy = self.moves[self.action]
    self.x = min(8, max(0, self.x + dx))
    self.y = min(8, max(0, self.y + dy))
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(frame == 0 and self.action == 5 and previous != 5)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _random_number(self):
    return int(self.random.uniform(1, 2.5))

  def _whack(self):
    for mole in self.moles:
      if not mole.timer and (mole.x, mole.y) == (self.x, self.y):
        if mole.value == self.target:
          self.target = self._random_number()
          for other in self.moles:
            other.timer, other.award = 60, other.value
        else:
          mole.timer, mole.award = 60, -mole.value
          self.penalty += 1
        break
    self.penalty += 1

  def _advance(self, whack):
    scroll = 0.02 + self.penalty
    self.penalty *= 0.9
    self.scroll += scroll
    if self.scroll > 0:
      self.numbers[1:] = self.numbers[:-1].copy()
      self.numbers[0] = self.random.integers(0, 10, 9)
      for mole in self.moles:
        mole.y += 1
      self.scroll -= 11
    if whack:
      self._whack()
    self.spawn_distance = self.spawn_distance - scroll if self.moles else -scroll
    if self.spawn_distance < 0:
      occupied = {m.x for m in self.moles if m.y == 0}
      for _ in range(9):
        x = int(self.random.integers(9))
        if x not in occupied:
          self.target_count -= 1
          value = self._random_number()
          if self.target_count <= 0:
            value = self.target
            self.target_count = int(self.random.integers(4, 7))
          self.moles.append(Mole(x, 0, value))
          break
      self.spawn_distance += self.random.uniform(9, 12)
    self.move_timer -= 1
    if self.move_timer < 0:
      self._move_moles()
      self.move_timer += 99
    remaining = []
    reward = 0
    for mole in self.moles:
      if mole.timer:
        mole.timer -= 1
        if not mole.timer:
          reward += mole.award
          continue
      elif mole.y >= 8:
        mole.timer = 60
        self.terminal |= mole.value == self.target
      remaining.append(mole)
    self.moles = remaining
    return reward

  def _move_moles(self):
    occupied = {(m.x, m.y) for m in self.moles}
    for mole in self.moles:
      if mole.timer:
        continue
      start = int(self.random.integers(4))
      for offset in range(4):
        dx, dy = self.mole_moves[(start + offset) % 4]
        x, y = mole.x + dx, mole.y + dy
        if 0 <= x < 9 and 0 < y < 9 and (x, y) not in occupied:
          occupied.remove((mole.x, mole.y))
          occupied.add((x, y))
          mole.x, mole.y = x, y
          break

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    visible = self.numbers.copy()
    for mole in self.moles:
      if mole.y < 9:
        if mole.timer:
          visible[mole.y, mole.x] = mole.value
          state[mole.y, mole.x, 4] = True
        else:
          visible[mole.y, mole.x] = (visible[mole.y, mole.x] + mole.value) % 10
    for bit in range(4):
      state[:9, :9, bit] = (visible & (1 << bit)) != 0
    state[self.y, self.x, 5] = True
    for col, value in enumerate((self.target, round(-self.scroll * 4), self.move_timer, self.action)):
      for bit in range(6):
        state[bit, 9, col] = bool(value & (1 << bit))
      state[9, col, 0] = bool(value & 64)
    return state
