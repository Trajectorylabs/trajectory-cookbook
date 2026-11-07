from dataclasses import dataclass

import numpy as np


@dataclass
class Card:
  column: int
  y: float
  rank: int
  opened: bool = False
  velocity: float = 0


class PairsDropEnvironment:
  score_cap = 200
  moves = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.cards: list[Card] = []
    self.opened: list[Card] = []
    self.column, self.row = 3, 4
    self.scroll = 0.0
    self.next_y, self.next_column = -6.0, 0
    self.timer = self.steps = self.action = 0
    self.multiplier = 1
    self.terminal = False
    self._deal()

  def _deal(self):
    for rank in self.random.permutation(np.tile(np.arange(1, 14), 2)):
      self.cards.append(Card(self.next_column, self.next_y, int(rank)))
      self.next_column = (self.next_column + 1) % 7
      if not self.next_column:
        self.next_y -= 12

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
    self.column = min(6, max(0, self.column + dx))
    self.row = min(9, max(0, self.row + dy))
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(frame == 0 and self.action == 5 and previous != 5)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _open(self):
    for card in self.cards:
      if (card.column == self.column and 0 <= card.y < 100
          and int(card.y // 10) == self.row and not card.opened and not card.velocity):
        if self.timer:
          self._close()
        card.opened = True
        self.opened.append(card)
        if len(self.opened) == 2:
          first, second = self.opened
          self.timer = 60
          if first.rank == second.rank:
            for other in self.cards:
              if any(other.column == match.column and other.y >= match.y for match in self.opened):
                other.velocity = 1
            self.opened = []
            self.multiplier = 1
        return True
    return False

  def _close(self):
    for card in self.opened:
      card.opened = False
    self.opened.clear()
    self.timer = 0

  def _advance(self, tap):
    if self.timer:
      self.timer -= 1
      if not self.timer:
        self._close()
    opened = tap and self._open()
    reward = 0
    remaining = []
    min_y, max_y = 99, 0
    for card in self.cards:
      card.y += self.scroll + card.velocity
      if card.velocity:
        card.velocity += 0.1
      min_y, max_y = min(min_y, card.y), max(max_y, card.y)
      if not card.velocity and card.y > 95:
        self.terminal = True
      if card.y > 99:
        reward += self.multiplier
        self.multiplier += 1
      else:
        remaining.append(card)
    self.cards = remaining
    self.next_y += self.scroll
    self.scroll += (0.0015 - self.scroll) * 0.2 + 0.15 * opened
    self.scroll += max(0, 50 - max_y) * 0.01
    if min_y > -6:
      self._deal()
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for card in self.cards:
      if 0 <= card.y < 100:
        row, col = int(card.y // 10), card.column
        state[row, col, 4] = True
        state[row, col, 5] |= card.velocity > 0
        if card.opened:
          for bit in range(4):
            state[row, col, bit] |= bool(card.rank & (1 << bit))
    state[self.row, 7, 0] = True
    state[self.column, 7, 1] = True
    values = (self.steps, self.timer, min(511, self.multiplier), self.action)
    for channel, value in enumerate(values):
      for bit in range(9):
        state[bit, 8, channel] = bool(value & (1 << bit))
    return state
