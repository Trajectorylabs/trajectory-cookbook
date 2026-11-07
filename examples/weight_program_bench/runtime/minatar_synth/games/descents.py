from dataclasses import dataclass
import math

import numpy as np


SHIP_X = 9


@dataclass
class Deck:
  x: float
  y: float
  width: float


class DescentsEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.y, self.vy = 9.0, 0.0
    self.up, self.down, self.next_up, self.next_down = -2, 2, -3, 3
    self.decks = [Deck(30, 50, 40), Deck(50, 75, 35), Deck(70, 99, 30)]

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
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    self.vy += (self.up if self.pressed else self.down) * 0.005
    self.y += self.vy
    scroll = (min(19, max(9, self.y)) - self.y) * 0.2
    self.y += scroll
    for deck in self.decks:
      deck.x -= 0.2
      deck.y += scroll
    first = self.decks[0]
    edge = first.x + first.width
    if self.y < 0 or edge < SHIP_X - 3:
      self.terminal = True
      return 0
    if first.x - 3 <= SHIP_X <= edge + 3 and abs(first.y - self.y) <= 2:
      if self.vy > 1:
        self.terminal = True
        return 0
      self.decks.pop(0)
      last = self.decks[-1]
      self.decks.append(Deck(last.x + last.width * 0.7 + self.random.uniform(0, 30),
                             last.y + self.random.uniform(20, 40), self.random.uniform(25, 50)))
      self.up, self.down = self.next_up, self.next_down
      self.next_up = math.floor(-4 + self.random.uniform(-3, 3))
      self.next_down = math.floor(4 + self.random.uniform(-2, 2))
      self.vy = 0
      return max(0, math.floor(edge - SHIP_X + 1))
    if self.y >= first.y + 1:
      self.terminal = True
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), 0, 0] = True
    for index, deck in enumerate(self.decks):
      if 0 <= deck.y < 100:
        row = int(deck.y) // 10
        for col in range(10):
          state[row, col, 1 if index == 0 else 2] |= col * 10 < deck.x + deck.width and (col + 1) * 10 > deck.x
    first = self.decks[0]
    if 0 <= first.y + 3 < 100:
      state[int(first.y + 3) // 10, :min(10, max(0, int((first.x + first.width) / 10) + 1)), 3] = True
    state[0, :3, 4] = self.pressed, self.vy > 0.6, self.vy > 1
    for row, value in enumerate((self.steps, round(self.vy * 20) + 256, -self.up,
                                 self.down, -self.next_up, self.next_down)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
