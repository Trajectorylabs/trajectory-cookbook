from dataclasses import dataclass

import numpy as np


@dataclass
class Pile:
  x: float
  y: float


class ZartanEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 99., 9.
    self.vx = self.vy = 0.
    self.piles = []
    self.anchor = self.nearest = None
    self.distance = 9.
    self.steps = self.action = 0
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
    pressed = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0.
    for frame in range(4):
      reward += self._advance(pressed and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, pressed):
    scroll = 0.1 + max(0, self.x - 30) * 0.1
    self.x -= scroll
    self.vy += 0.02
    if self.y < 0 and self.vy < 0:
      self.vy *= -1
    if self.y > 99:
      self.terminal = True
      return scroll
    self.vx *= 0.99
    self.vy *= 0.99
    self.x += self.vx
    self.y += self.vy

    self.nearest = min(
      (pile for pile in self.piles if pile.x > self.x and abs(pile.y - self.y) < 99),
      key=lambda pile: abs(pile.y - self.y),
      default=self.nearest,
    )
    if pressed:
      self.anchor = self.nearest
    if self.action != 5:
      self.anchor = None
    elif self.anchor is not None:
      self.vx += (self.anchor.x - self.x) / 199
      self.vy += (self.anchor.y - self.y) / 199
      if self.anchor.x < 0:
        self.anchor = None

    self.distance -= scroll
    if self.distance < 0:
      self.distance += self.random.uniform(9, 66)
      self.piles.append(Pile(99, self.random.uniform(0, 66)))
    for pile in self.piles:
      pile.x -= scroll
    self.piles = [pile for pile in self.piles if pile.x > 0]
    return scroll

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    _mark(state, self.x, self.y, 0)
    for pile in self.piles:
      _mark(state, pile.x, pile.y, 1)
    if self.nearest is not None:
      _mark(state, self.nearest.x, self.nearest.y, 2)
    if self.anchor is not None:
      _mark(state, self.anchor.x, self.anchor.y, 3)
      for x, y in zip(np.linspace(self.x, self.anchor.x, 41),
                      np.linspace(self.y, self.anchor.y, 41), strict=True):
        _mark(state, x, y, 4)
    values = (self.steps, self.action, round((self.vx + 8) * 16),
              round((self.vy + 8) * 16), round(self.distance * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _mark(state, x, y, channel):
  state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), channel] = True
